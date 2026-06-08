from .smaccl.smaccl import Smaccl
import numpy as np
#from .c3s_lib import set_ac_flag, closest_model_vito, pre_brdf, Ps, dPsdzQ, closest_models
from .c3s_lib import set_ac_flag, Ps, closest_models, dPsdz, closest_model
import math
import xarray as xa
import dask.array as da
from .KG_AOD import get_aod_uncertainty_multiplier, KoppenGeiger

__module__    = "ISmaccl.py"    
__version__   = "1.01.00"

#def rsme_aod(tau_550, sigma_base, sigma_rel):
#    return np.maximum(sigma_base, sigma_rel*tau_550)

class ISmaccl(object):
    """
    ISmaccl is a wrapper for the Smaccl class that provides an interface
    for initializing and using the Smaccl library.
    """

    def __init__(self, config, frac_aer_model, ca, XBLOCK=128, XGRID=128, breakpoint=False, ancillary=False, *args, **kwargs):
        """
        Initialize the ISmaccl instance with the provided arguments.
        """
#        self.smaccl = Smaccl(*args, **kwargs)
        self.smaccl = Smaccl(platform=kwargs['platform'])
        self.config = config

        # SMACG configuration
        self.XBLOCK = XBLOCK
        self.XGRID = XGRID
        self.NBLOOP = 1
        self.breakpoint = breakpoint
        self.ancillary = ancillary
        self.frac_aer_model = frac_aer_model
        self.ca = ca
        self.Nmodels = config['nmodels']
        self.map_file = config['kp_map_file']
        self.legend_file = config['legend_file']
        self.mapping_file = config['mapping_file']

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying Smaccl instance.
        """
        return getattr(self.smaccl, name)
    
    def _convert_to_numpy(self, arr, good=None):
        """Convert dask or numpy array to numpy array with proper type."""
        if hasattr(arr, 'values'):  # xarray DataArray
            arr = arr.values

        if good is not None:
            arr = arr[good]

        # Convert to float32 with C order
        return np.asarray(arr, dtype=np.float32, order='C')
    
    def run_block(self, l2_data, batch_size=512):
        ds_in = xa.Dataset(
        {
            x: l2_data[x] for x in ['lat','lon','TOA','SZA','VZA','SAA','VAA','VZA_IR','VAA_IR','TQV','TO3','TOTEXTTAU','SLP','elev','Delev','T10M','SU_FRAC','SS_FRAC','DU_FRAC','OC_FRAC','BC_FRAC','ERROR']
        }
        )
        ds_in = ds_in.assign_attrs(l2_data.attrs)
        y_chunk = ds_in['SZA'].chunks[0][0] if hasattr(ds_in['SZA'], 'chunks') else batch_size
        x_chunk = ds_in['SZA'].chunks[1][0] if hasattr(ds_in['SZA'], 'chunks') else batch_size
#        template_4d = l2_data['TOA'].expand_dims({'aermodel':self.Nmodels}, axis=3).chunk({'bands':-1, 'y':y_chunk, 'x':x_chunk, 'aermodel':-1})
        template_3d = l2_data['TOA'].chunk({'bands':-1, 'y':y_chunk, 'x':x_chunk})
        template = xa.Dataset({'rTOC': template_3d,
                               'UrTOC': template_3d,
                               'Jrtoa': template_3d,
                               'Juh2o': template_3d,
                               'Juo3': template_3d,
                               'Jpre': template_3d,
                               'Drsurf': template_3d,
                            }).chunk({'bands':-1, 'y':y_chunk, 'x':x_chunk, 'aermodel':-1})
        ds_out = xa.map_blocks(
            self.run,
            ds_in, 
            template=template,
        )
        return ds_out
            
    def run(self, l2_data, iaero):
                
        """
        Run the ISmaccl instance with the provided arguments.
        """

        sza = l2_data['SZA'].values
        SIZE1, SIZE2 = sza.shape
        good = np.where(sza < 90)

        if good[0].size == 0:
            t_2d = l2_data['SZA'] + np.nan   
            t_3d = l2_data['TOA'] + np.nan
            ds_out = xa.Dataset(
                {
#                    'rsurf': t_4d,
                    'rTOC': t_3d,
                    'rTOC_0': t_3d,
                    'UrTOC': t_3d,
                    'Jrtoa': t_3d,
                    'Juh2o': t_3d,
                    'Juo3': t_3d,
                    'Jpre': t_3d,
                    'Drsurf': t_3d,
                    'Duh2o': t_3d,
                    'Duo3': t_3d,
                    'Drtoa': t_3d,
                    'Dpre': t_3d,
                    'Dtaup': t_3d,
                    'UrTOC_ens' : t_3d,
                    'Jtau550' : t_3d
                },
            )
            return ds_out

        sza = sza[good]
        SIZE = sza.shape[0]
        saa = l2_data['SAA'].values[good]
        vza = l2_data['VZA'].values[good]
        taup550 = l2_data['TOTEXTTAU'].values[good]
        uh2o = l2_data['TQV'].values[good]
        uo3 = l2_data['TO3'].values[good]
        p0 = l2_data['SLP'].values[good]
        t10m = l2_data['T10M'].values[good]
        alt = l2_data['elev'].values[good]
        Dalt = l2_data['Delev'].values[good]
        lat = l2_data['lat'].values[good]
        lon = l2_data['lon'].values[good]
        match = {'sulf':'SU_FRAC', 'dust':'DU_FRAC', 'oc':'OC_FRAC', 'ssalt':'SS_FRAC', 'bc':'BC_FRAC'}
        frac = np.zeros((len(match), SIZE), dtype='float32')
        for k,key in enumerate(self.frac_aer_model.keys()):
            frac[k] = l2_data[match[key]].values[good]
        band_err = l2_data['ERROR'].values[:, good[0], good[1]]
        k1p = l2_data['k1p'].values[:, good[0], good[1]]
        k2p = l2_data['k2p'].values[:, good[0], good[1]]

        flag = (taup550 > self.config['taot'])
        ac_process_flag = np.zeros((SIZE1, SIZE2), dtype='ubyte')
        ac_process_flag[good] = flag.astype('u1')

        climato = False
        ac_flag = np.zeros((SIZE1, SIZE2), dtype='uint32')
        ac_flag[good] = set_ac_flag(taup550, sza, vza, climato)
        
        iaero_good = iaero[:, good[0], good[1]]

        rsurf = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan 
        rsurf0 = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        dev_std = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Jrtoa = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Juh2o = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Juo3 = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Jpre = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Jtaup = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Duh2o = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Duo3 = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Drtoa = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Dpre = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Dtaup = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        if l2_data.sensor == 'Proba-V':
            band_data = np.zeros((2, SIZE1, SIZE2), dtype='float32')
            band_data = l2_data['TOA'].values[:2, good[0], good[1]]
            vza = l2_data['VZA'].values[good]
            vaa = l2_data['VAA'].values[good]
            toc_data_vis = self.exec_smaccl(sza, vza, saa, vaa, taup550, uh2o, uo3, p0, t10m, alt, Dalt, lat, lon, band_data, band_err, frac, self.frac_aer_model, self.ca[:2], iaero_good, k1p, k2p)
            rsurf[:2, good[0], good[1]] = toc_data_vis[0]
            rsurf0[:2, good[0], good[1]] = toc_data_vis[1]
            dev_std[:2, good[0], good[1]] = toc_data_vis[2]
            Juh2o[:2, good[0], good[1]] = toc_data_vis[3]
            Juo3[:2, good[0], good[1]] = toc_data_vis[4]
            Jrtoa[:2, good[0], good[1]] = toc_data_vis[5]
            Jpre[:2, good[0], good[1]] = toc_data_vis[6]
            Jtaup[:2, good[0], good[1]] = toc_data_vis[7]
            Duh2o[:2, good[0], good[1]] = toc_data_vis[8]
            Duo3[:2, good[0], good[1]] = toc_data_vis[9]
            Drtoa[:2, good[0], good[1]] = toc_data_vis[10]
            Dpre[:2, good[0], good[1]] = toc_data_vis[11]
            Dtaup[:2, good[0], good[1]] = toc_data_vis[12]

            band_data = l2_data['TOA'].values[2:, good[0], good[1]]
            vza = l2_data['VZA_IR'].values[good]
            vaa = l2_data['VAA_IR'].values[good]
            toc_data_ir = self.exec_smaccl(sza, vza, saa, vaa, taup550, uh2o, uo3, p0, t10m, alt, Dalt, lat, lon, band_data, band_err, frac, self.frac_aer_model, self.ca[2:], iaero_good, k1p, k2p)
            rsurf[2:, good[0], good[1]] = toc_data_ir[0]
            rsurf0[2:, good[0], good[1]] = toc_data_ir[1]
            dev_std[2:, good[0], good[1]] = toc_data_ir[2]
            Juh2o[2:, good[0], good[1]] = toc_data_ir[3]
            Juo3[2:, good[0], good[1]] = toc_data_ir[4]
            Jrtoa[2:, good[0], good[1]] = toc_data_ir[5]
            Jpre[2:, good[0], good[1]] = toc_data_ir[6]
            Jtaup[2:, good[0], good[1]] = toc_data_ir[7]
            Duh2o[2:, good[0], good[1]] = toc_data_ir[8]
            Duo3[2:, good[0], good[1]] = toc_data_ir[9]
            Drtoa[2:, good[0], good[1]] = toc_data_ir[10]
            Dpre[2:, good[0], good[1]] = toc_data_ir[11]
            Dtaup[2:, good[0], good[1]] = toc_data_ir[12]

        else:
            band_data = np.zeros((len(l2_data.bands), SIZE1, SIZE2), dtype='float32')
            band_data = l2_data['TOA'].values[:,good[0], good[1]]
            vza = l2_data['VZA'].values[good]
            vaa = l2_data['VAA'].values[good]
            toc_data = self.exec_smaccl(sza, vza, saa, vaa, taup550, uh2o, uo3, p0, t10m, alt, Dalt, lat, lon, band_data, band_err, frac, self.frac_aer_model, self.ca, iaero_good, k1p, k2p)
            rsurf[:, good[0], good[1]] = toc_data[0]
            rsurf0[:, good[0], good[1]] = toc_data[1]
            dev_std[:, good[0], good[1]] = toc_data[2]
            Juh2o[:, good[0], good[1]] = toc_data[3]
            Juo3[:, good[0], good[1]] = toc_data[4]
            Jrtoa[:, good[0], good[1]] = toc_data[5]
            Jpre[:, good[0], good[1]] = toc_data[6]
            Jtaup[:, good[0], good[1]] = toc_data[7]
            Duh2o[:, good[0], good[1]] = toc_data[8]
            Duo3[:, good[0], good[1]] = toc_data[9]
            Drtoa[:, good[0], good[1]] = toc_data[10]
            Dpre[:, good[0], good[1]] = toc_data[11]
            Dtaup[:, good[0], good[1]] = toc_data[12]


#        t_4d = (l2_data['TOA'].expand_dims({'aermodel':iaero_good.shape[0]}, axis=3)) + np.nan
        t_2d = l2_data['SZA'] + np.nan
        t_2d.data = ac_process_flag
        t_3d = l2_data['TOA'] + np.nan
#        t_4d.data = rsurf
        t_3d.data = rsurf
        ds_out = xa.Dataset({'rTOC': t_3d,
#                             'ac_process_flag': t_2d,
                        }
        )
        t_3d.data = rsurf0
        ds_out['rTOC_0'] = t_3d
        t_3d.data = dev_std
        ds_out['UrTOC_ens'] = t_3d
        t_3d.data = Jrtoa
        ds_out['Jrtoa'] = t_3d
        t_3d.data = Juh2o
        ds_out['Juh2o'] = t_3d
        t_3d.data = Juo3
        ds_out['Juo3'] = t_3d
        t_3d.data = Jpre
        ds_out['Jpre'] = t_3d
        t_3d.data = Jtaup
        ds_out['Jtau550'] = t_3d
        t_3d.data = Duh2o
        ds_out['Duh2o'] = t_3d
        t_3d.data = Duo3
        ds_out['Duo3'] = t_3d
        t_3d.data = Drtoa
        ds_out['Drtoa'] = t_3d
        t_3d.data = Dpre
        ds_out['Dpre'] = t_3d
        t_3d.data = Dtaup
        ds_out['Dtaup'] = t_3d

        return ds_out

    def exec_smaccl(self,  sza, vza, saa, vaa, taup550, uh2o, uo3, p0, t10m, alt, Dalt, lat, lon, band_data, band_err, frac, frac_aer_model, coeffs, iaero, k1p, k2p):
        NB = band_data.shape[0]


        filtre = np.isnan(lat)
        lat[filtre] = 0
        lon[filtre] = 0
        # aerosol model computation
#        xb = []
#        xm = []
#        if 'aero_nmod' in self.config.keys():
#            nb_pixel = len(lat)
#            iaero2 = np.zeros(nb_pixel)
#            iaero2[:] = self.config['aero_nmod']
#        else:
#            for k,key in enumerate(frac_aer_model.keys()):
#                xb.append(frac_aer_model[key])
#                xm.append(frac[k])
#            xb = np.stack(xb, axis=0)
#            xm = np.stack(xm, axis=0)
#            iaero2 = closest_models(xm, xb, self.Nmodels)
#        print("iaero models: ", iaero2.shape)
        Naero = len(iaero)

#        # flag large aot pixels
        GSIZE = sza.shape[0]
#        flag = (taup550 > self.config['taot'])
#        ac_process_flag = np.zeros((SIZE1, SIZE2), dtype='ubyte')
#        ac_process_flag = np.zeros((GSIZE), dtype='ubyte')
#        ac_process_flag[good] = flag.astype('u1')
#        ac_process_flag = flag.astype('u1')
#        climato = False
#        ac_flag = np.zeros((SIZE1, SIZE2), dtype='uint32')
#        ac_flag[good] = set_ac_flag(taup550, sza, vza, climato)
#        ac_flag = np.zeros((GSIZE), dtype='uint32')
#        ac_flag = set_ac_flag(taup550, sza, vza, climato)


#        GSIZE= int(good[0].size)
        # brdf arrays
        # Test for BRDF input data for correction
#        if brdf is None:
#            k1p = np.zeros((NB, GSIZE), dtype=np.float32, order='C')
#            k2p = np.zeros((NB, GSIZE), dtype=np.float32, order='C')
#        else:
#        print("BRDF inputs for correction: {}".format(self.config['brdf']))
#        k2p = brdf['kp12'].data[1].astype(np.float32, order='C')
#        k1p = brdf['kp12'].data[0].astype(np.float32, order='C')


#        k_uh2o = self.config['k_uh2o']
#        k_uo3 =  self.config['k_uo3']
#        k_p0 =   self.config['k_p0']
        Epre = self.config['epre']
        Etaup =  self.config['etaup']
        ERtaup = self.config['ertaup']
        Euo3 =   self.config['euo3']
        ERuo3 =  self.config['eruo3']
        Euh2o =  self.config['euh2o']
        ERuh2o = self.config['eruh2o']
        # pressure correction for surface altitude and transformation from Pa to hPa

#        pressure = Ps(alt, p0*k_p0, t10m)
        pressure = Ps(alt, p0, t10m)
        # quadratic mean of error due to met fields (Epre) and error due to altitude (Dalt)
#        pressure_err = np.sqrt((dPsdz(alt, p0*k_p0, t10m) * Dalt)**2 + Epre**2)/2.
        pressure_err = np.sqrt((dPsdz(alt, p0, t10m) * Dalt)**2 + Epre**2)/2.
        # conversion from kg.m-2 to g.cm-2
#        uh2o *= k_uh2o
        # conversion from Dobson to cm.atm 
#        uo3  *= k_uo3

        # prepare radiometry array
#        rtoa = band_data[:, good[0], good[1]]
#        rtoa_err = band_err[:, good[0], good[1]]
        rtoa = band_data[:,:]
        rtoa_err = band_err[:,:]

        Z = int(math.ceil(float(GSIZE)/float(self.XBLOCK*self.XGRID)))
        GSIZEXT = Z * self.XBLOCK * self.XGRID

        # the \"ext\" suffix is for extended arrays, larger than the good pixels size, it is completed by NaN's\n",
        rtoa_ext     = np.zeros((NB, GSIZEXT), dtype='float32') + np.nan
        k1p_ext      = np.zeros((NB, GSIZEXT), dtype='float32') + np.nan
        k2p_ext      = np.zeros((NB, GSIZEXT), dtype='float32') + np.nan
        taup550_ext  = np.zeros((GSIZEXT), dtype='float32') + np.nan                                                                                                               
        uo3_ext      = np.zeros((GSIZEXT), dtype='float32') + np.nan
        pressure_ext = np.zeros((GSIZEXT), dtype='float32') + np.nan
        uh2o_ext     = np.zeros((GSIZEXT), dtype='float32') + np.nan
        tetas_ext    = np.zeros((GSIZEXT), dtype='float32') + np.nan
        tetav_ext    = np.zeros((GSIZEXT), dtype='float32') + np.nan
        phis_ext     = np.zeros((GSIZEXT), dtype='float32') + np.nan
        phiv_ext     = np.zeros((GSIZEXT), dtype='float32') + np.nan
        iaero_ext    = np.zeros((Naero, GSIZEXT), dtype='int16')

        for i in range(NB):
            rtoa_ext[i,:GSIZE]= rtoa[i,:]
            k1p_ext[i,:GSIZE] = k1p[i,:]
            k2p_ext[i,:GSIZE] = k2p[i,:]

        taup550_ext[:GSIZE]  = taup550
        uo3_ext[:GSIZE]      = uo3
        pressure_ext[:GSIZE] = pressure
        uh2o_ext[:GSIZE]     = uh2o
        tetas_ext[:GSIZE]    = sza
        tetav_ext[:GSIZE]    = vza
        phis_ext[:GSIZE]     = saa
        phiv_ext[:GSIZE]     = vaa
        iaero_ext[:,:GSIZE]    = iaero

        # Input arrays reshaping
        k1p_ext      = np.reshape(k1p_ext,  (NB,Z,self.XBLOCK,self.XGRID), order='C')
        k2p_ext      = np.reshape(k2p_ext,  (NB,Z,self.XBLOCK,self.XGRID), order='C')
        rtoa_ext     = np.reshape(rtoa_ext, (NB,Z,self.XBLOCK,self.XGRID), order='C')
        tetas_ext    = np.reshape(tetas_ext,   (Z,self.XBLOCK,self.XGRID),    order='C')
        tetav_ext    = np.reshape(tetav_ext,   (Z,self.XBLOCK,self.XGRID),    order='C')
        phis_ext     = np.reshape(phis_ext,    (Z,self.XBLOCK,self.XGRID),    order='C')
        phiv_ext     = np.reshape(phiv_ext,    (Z,self.XBLOCK,self.XGRID),    order='C')
        uh2o_ext     = np.reshape(uh2o_ext,    (Z,self.XBLOCK,self.XGRID),    order='C')
        uo3_ext      = np.reshape(uo3_ext,     (Z,self.XBLOCK,self.XGRID),    order='C')
        taup550_ext  = np.reshape(taup550_ext, (Z,self.XBLOCK,self.XGRID),    order='C')
        pressure_ext = np.reshape(pressure_ext,(Z,self.XBLOCK,self.XGRID),    order='C')
        iaero_ext    = np.reshape(iaero_ext   ,(Naero, Z,self.XBLOCK,self.XGRID),    order='C')

        (rsurf_ext, rsurf0_ext, dev_std_ext, Jrtoa_ext, Juo3_ext,Juh2o_ext,Jpre_ext,Jtaup_ext) = self.smaccl.run(
                coeffs, tetas_ext, tetav_ext,phis_ext, phiv_ext, uh2o_ext, uo3_ext, 
                taup550_ext, pressure_ext, rtoa_ext, k1p_ext,
                k2p_ext, iaero_ext, 
                NBLOOP=self.NBLOOP)
#                XBLOCK=self.XBLOCK, XGRID=self.XGRID, NBLOOP=self.NBLOOP)

        # Output arrays reshaping
#        rsurf_ext = np.reshape(rsurf_ext,(Naero, NB,GSIZEXT), order='C')
        rsurf_ext = np.reshape(rsurf_ext,(NB,GSIZEXT), order='C')
        rsurf0_ext = np.reshape(rsurf0_ext,(NB,GSIZEXT), order='C')
        dev_std_ext = np.reshape(dev_std_ext,(NB,GSIZEXT), order='C')
        Jrtoa_ext = np.reshape(Jrtoa_ext,(NB,GSIZEXT), order='C')
        Juo3_ext  = np.reshape(Juo3_ext, (NB,GSIZEXT), order='C')
        Juh2o_ext = np.reshape(Juh2o_ext,(NB,GSIZEXT), order='C')
        Jpre_ext  = np.reshape(Jpre_ext, (NB,GSIZEXT), order='C')
        Jtaup_ext = np.reshape(Jtaup_ext,(NB,GSIZEXT), order='C')


        rsurf = np.zeros((NB,GSIZE), dtype=np.float32)
        rsurf0 = np.zeros((NB,GSIZE), dtype=np.float32)
        dev_std = np.zeros((NB,GSIZE), dtype=np.float32)
        Jrtoa  = np.zeros((NB,GSIZE), dtype=np.float32)
        Juo3   = np.zeros((NB,GSIZE), dtype=np.float32)
        Juh2o  = np.zeros((NB,GSIZE), dtype=np.float32)
        Jpre   = np.zeros((NB,GSIZE), dtype=np.float32)
        Jtaup  = np.zeros((NB,GSIZE), dtype=np.float32)
        Drtoa  = np.zeros((NB,GSIZE), dtype=np.float32)
        Duo3   = np.zeros((NB,GSIZE), dtype=np.float32)
        Duh2o  = np.zeros((NB,GSIZE), dtype=np.float32)
        Dpre   = np.zeros((NB,GSIZE), dtype=np.float32)
        Dtaup  = np.zeros((NB,GSIZE), dtype=np.float32)

        if self.ancillary:
            Iuo3   = np.zeros((GSIZE), dtype=np.float32) + np.nan
            Iuh2o  = np.zeros((GSIZE), dtype=np.float32) + np.nan
            Ipre   = np.zeros((GSIZE), dtype=np.float32) + np.nan
            Itaup  = np.zeros((GSIZE), dtype=np.float32) + np.nan
            Ialt   = np.zeros((GSIZE), dtype=np.float32) + np.nan
            Iaero  = np.zeros((GSIZE), dtype='int32')
            Iuo3   = uo3
            Iuh2o  = uh2o
            Ipre   = pressure
            Itaup  = taup550
            Ialt   = alt
            Iaero  = iaero
            ancillary    = [Iuo3,Iuh2o,Itaup,Iaero,Ipre,Ialt]

        for i in range(NB):
#            for j in range(Naero):
#            inter  = rsurf_ext[i,:GSIZE]
#                rsurf[i,:,:,j] = inter[:,:]
            rsurf[i,:] = rsurf_ext[i,:GSIZE]
            rsurf0[i,:] = rsurf0_ext[i,:GSIZE]
#            Drsurf[i,:]= inter
#            inter  = dev_std_ext[i,:GSIZE]
#            stock = inter*inter
            dev_std[i,:] = dev_std_ext[i,:GSIZE]

#            inter  = abs(Jrtoa_ext[i,:GSIZE]  * rtoa_err[i,:]               ) # it is 0 because no input error
#            inter_drtoa = rtoa_err[i,:]
            Drtoa[i,:] = rtoa_err[i,:]
            Jrtoa[i,:] = Jrtoa_ext[i,:GSIZE]
#            stock        += inter**2

#            inter_dtaup = rsme_aod(taup550, Etaup, ERtaup)
#            inter  = abs(Jtaup_ext[i,:GSIZE] * inter_dtaup)
#            Dtaup[i,:] = rsme_aod(taup550, Etaup, ERtaup)
            _kg_instance = KoppenGeiger(map_file=self.map_file, legend_file=self.legend_file, mapping_file=self.mapping_file)
            mults = get_aod_uncertainty_multiplier(lat,lon, _kg_instance)
            Dtaup[i,:] = mults*np.maximum(Etaup, taup550*ERtaup)
            Jtaup[i,:] = Jtaup_ext[i,:GSIZE]
#            stock       += inter**2

#            inter  = abs(Juo3_ext[i,:GSIZE]  * (Euo3  + ERuo3  * uo3      ))
            Duo3[i,:]  = Euo3  + ERuo3  * uo3
            Juo3[i,:]  = Juo3_ext[i,:GSIZE]
#            stock       += inter**2
#            inter  = abs(Juh2o_ext[i,:GSIZE] * (Euh2o + ERuh2o * uh2o     ))
            #if self.breakpoint: 
#            Duh2o[i,:,:] = inter
            Duh2o[i,:] = Euh2o + ERuh2o * uh2o
            Juh2o[i,:] = Juh2o_ext[i,:GSIZE]
#            stock       += inter**2
#            inter  = abs(Jpre_ext[i,:GSIZE] *  pressure_err)
            #if self.breakpoint: 
#            Dpre[i,:,:]  = inter
            Dpre[i,:]  = pressure_err
            Jpre[i,:]  = Jpre_ext[i,:GSIZE]
#            stock       += inter**2

#            inter  = np.sqrt(stock)
#            Drsurf[i,:,:]= inter

#            if self.breakpoint:
#            inter  = Jrtoa_ext[i,:GSIZE]
#            Jrtoa[i,:,:] = inter
#            Jrtoa[i,:] = inter
#            inter  = Juo3_ext[i,:GSIZE]
#            Juo3[i,:,:]  = inter
#            inter  = Juh2o_ext[i,:GSIZE]
#            Juh2o[i,:,:] = inter
#            inter  = Jpre_ext[i,:GSIZE]
#            Jpre[i,:,:]  = inter
#            inter  = Jtaup_ext[i,:GSIZE]
#            Jtaup[i,:,:] = inter

#        return rsurf, dev_std, Juh2o, Juo3, Jrtoa, Jpre, Jtaup, Drsurf, Duh2o, Duo3, Drtoa, Dpre, Dtaup
        return rsurf, rsurf0, dev_std, Juh2o, Juo3, Jrtoa, Jpre, Jtaup, Duh2o, Duo3, Drtoa, Dpre, Dtaup
#        return rsurf_ext, rsurf0_ext, dev_std_ext, Juh2o, Juo3, Jrtoa, Jpre, Jtaup, Duh2o, Duo3, Drtoa, Dpre, Dtaup