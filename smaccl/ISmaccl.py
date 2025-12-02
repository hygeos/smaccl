from smaccl import Smaccl
import numpy as np
#from .c3s_lib import set_ac_flag, closest_model_vito, pre_brdf, Ps, dPsdzQ, closest_models
from .c3s_lib import set_ac_flag, Ps, closest_models, dPsdz, closest_model
import math
import xarray as xa
import dask.array as da

__module__    = "ISmaccl.py"    
__version__   = "1.01.00"

class ISmaccl(object):
    """
    ISmaccl is a wrapper for the Smaccl class that provides an interface
    for initializing and using the Smaccl library.
    """

    def __init__(self, config, frac_aer_model, ca, breakpoint=False, ancillary=False, *args, **kwargs):
        """
        Initialize the ISmaccl instance with the provided arguments.
        """
        self.smaccl = Smaccl(*args, **kwargs)
        self.config = config

        # SMACG configuration
        self.XBLOCK = 256
        self.XGRID = 256
        self.NBLOOP = 1
        self.breakpoint = breakpoint
        self.ancillary = ancillary
        self.frac_aer_model = frac_aer_model
        self.ca = ca

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
        template_4d = l2_data['TOA'].expand_dims({'aermodel':11}, axis=3).chunk({'bands':-1, 'y':y_chunk, 'x':x_chunk, 'aermodel':-1})
        template_3d = l2_data['TOA'].chunk({'bands':-1, 'y':y_chunk, 'x':x_chunk})
        template = xa.Dataset({'rsurf': template_4d,
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
            
    def run(self, l2_data, brdf=None):
                
        """
        Run the ISmaccl instance with the provided arguments.
        """

        sza = l2_data['SZA'].values
        SIZE1, SIZE2 = sza.shape
        good = np.where(sza < 90)
        if good[0].size == 0:
            t_4d = (l2_data['TOA'].expand_dims({'aermodel':11}, axis=3)) + np.nan
            t_3d = l2_data['TOA'] + np.nan
            ds_out = xa.Dataset(
                {
                    'rsurf': t_4d,
                    'Jrtoa': t_3d,
                    'Juh2o': t_3d,
                    'Juo3': t_3d,
                    'Jpre': t_3d,
                    'Drsurf': t_3d,
                },
            )
            return ds_out

        sza = sza[good]
        SIZE = sza.shape[0]
        saa = l2_data['SAA'].values[good]
        taup550 = l2_data['TOTEXTTAU'].values[good]
        uh2o = l2_data['TQV'].values[good]
        uo3 = l2_data['TO3'].values[good]
        p0 = l2_data['SLP'].values[good]
        t10m = l2_data['T10M'].values[good]
        alt = l2_data['elev'].values[good]
        Dalt = l2_data['Delev'].values[good]
        lat = l2_data['lat'].values[good]
        match = {'sulf':'SU_FRAC', 'dust':'DU_FRAC', 'oc':'OC_FRAC', 'ssalt':'SS_FRAC', 'bc':'BC_FRAC'}
        frac = np.zeros((len(match), SIZE), dtype='float32')
        for k,key in enumerate(self.frac_aer_model.keys()):
            frac[k] = l2_data[match[key]].values[good]
        band_err = l2_data['ERROR'].values[:, good[0], good[1]]

        rsurf = np.zeros((4, SIZE1, SIZE2, 11), dtype='float32') + np.nan 
        Jrtoa = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Juh2o = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Juo3 = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Jpre = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan
        Drsurf = np.zeros((4, SIZE1, SIZE2), dtype='float32') + np.nan

        if l2_data.sensor == 'Proba-V':
            band_data = np.zeros((2, SIZE1, SIZE2), dtype='float32')
            band_data = l2_data['TOA'].values[:2, good[0], good[1]]
            vza = l2_data['VZA'].values[good]
            vaa = l2_data['VAA'].values[good]
            toc_data_vis = self.exec_smaccl(sza, vza, saa, vaa, taup550, uh2o, uo3, p0, t10m, alt, Dalt, lat, None, band_data, band_err, frac, self.frac_aer_model, self.ca[:2], brdf)
            rsurf[:2, good[0], good[1], :] = toc_data_vis[0]
            Juh2o[:2, good[0], good[1]] = toc_data_vis[1]
            Juo3[:2, good[0], good[1]] = toc_data_vis[2]
            Jrtoa[:2, good[0], good[1]] = toc_data_vis[3]
            Jpre[:2, good[0], good[1]] = toc_data_vis[4]
            Drsurf[:2, good[0], good[1]] = toc_data_vis[5]

            band_data = l2_data['TOA'].values[2:, good[0], good[1]]
            vza = l2_data['VZA_IR'].values[good]
            vaa = l2_data['VAA_IR'].values[good]
            toc_data_ir = self.exec_smaccl(sza, vza, saa, vaa, taup550, uh2o, uo3, p0, t10m, alt, Dalt, lat, None, band_data, band_err, frac, self.frac_aer_model, self.ca[2:], brdf)
            rsurf[2:, good[0], good[1], :] = toc_data_ir[0]
            Juh2o[2:, good[0], good[1]] = toc_data_ir[1]
            Juo3[2:, good[0], good[1]] = toc_data_ir[2]
            Jrtoa[2:, good[0], good[1]] = toc_data_ir[3]
            Jpre[2:, good[0], good[1]] = toc_data_ir[4]
            Drsurf[2:, good[0], good[1]] = toc_data_ir[5]

        else:
            band_data = np.zeros((len(l2_data.bands), SIZE1, SIZE2), dtype='float32')
            band_data = l2_data['TOA'].values[:,good[0], good[1]]
            vza = l2_data['VZA'].values[good]
            vaa = l2_data['VAA'].values[good]
            toc_data = self.exec_smaccl(sza, vza, saa, vaa, taup550, uh2o, uo3, p0, t10m, alt, Dalt, lat, None, band_data, band_err, frac, self.frac_aer_model, self.ca, brdf)
            rsurf[:, good[0], good[1], :] = toc_data[0]
            Juh2o[:, good[0], good[1]] = toc_data[1]
            Juo3[:, good[0], good[1]] = toc_data[2]
            Jrtoa[:, good[0], good[1]] = toc_data[3]
            Jpre[:, good[0], good[1]] = toc_data[4]
            Drsurf[:, good[0], good[1]] = toc_data[5]


        t_4d = (l2_data['TOA'].expand_dims({'aermodel':11}, axis=3)) + np.nan
        t_3d = l2_data['TOA'] + np.nan
        t_4d.data = rsurf
        t_3d.data = Jrtoa
        ds_out = xa.Dataset({'rsurf': t_4d,
                            'Jrtoa': t_3d,
                        }
        )
        t_3d.data = Juh2o
        ds_out['Juh2o'] = t_3d
        t_3d.data = Juo3
        ds_out['Juo3'] = t_3d
        t_3d.data = Jpre
        ds_out['Jpre'] = t_3d
        t_3d.data = Drsurf
        ds_out['Drsurf'] = t_3d
        return ds_out

    def exec_smaccl(self,  sza, vza, saa, vaa, taup550, uh2o, uo3, p0, t10m, alt, Dalt, lat, lon, band_data, band_err, frac, frac_aer_model, coeffs, brdf=None):
        NB = band_data.shape[0]

        # aerosol model computation
        xb = []
        xm = []
        if 'aero_nmod' in self.config.keys():
            nb_pixel = len(lat)
            iaero = np.zeros(nb_pixel)
            iaero[:] = self.config['aero_nmod']
        else:
            for k,key in enumerate(frac_aer_model.keys()):
                xb.append(frac_aer_model[key])
                xm.append(frac[k])
            xb = np.stack(xb, axis=0)
            xm = np.stack(xm, axis=0)
            iaero = closest_models(xm, xb)
        Naero = len(iaero)

#        # flag large aot pixels
        GSIZE = sza.shape[0]
        flag = (taup550 > self.config['taot'])
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
        if brdf is None:
            k1p = np.zeros((NB, GSIZE), dtype=np.float32, order='C')
            k2p = np.zeros((NB, GSIZE), dtype=np.float32, order='C')
        else:
            print("BRDF inputs for correction: {}".format(self.config['brdf']))
            k2p = brdf['kp12'].data[1].astype(np.float32, order='C')
            k1p = brdf['kp12'].data[0].astype(np.float32, order='C')


        k_uh2o = self.config['k_uh2o']
        k_uo3 =  self.config['k_uo3']
        k_p0 =   self.config['k_p0']
        Epre = self.config['epre']
        Etaup =  self.config['etaup']
        ERtaup = self.config['ertaup']
        Euo3 =   self.config['euo3']
        ERuo3 =  self.config['eruo3']
        Euh2o =  self.config['euh2o']
        ERuh2o = self.config['eruh2o']
        # pressure correction for surface altitude and transformation from Pa to hPa

        pressure = Ps(alt, p0*k_p0, t10m)
        # quadratic mean of error due to met fields (Epre) and error due to altitude (Dalt)
        pressure_err = np.sqrt((dPsdz(alt, p0*k_p0, t10m) * Dalt)**2 + Epre**2)/2.
        # conversion from kg.m-2 to g.cm-2
        uh2o *= k_uh2o
        # conversion from Dobson to cm.atm 
        uo3  *= k_uo3

        # prepare radiometry array
#        rtoa = band_data[:, good[0], good[1]]
#        rtoa_err = band_err[:, good[0], good[1]]
        rtoa = band_data
        rtoa_err = band_err

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

        (rsurf_ext,Jrtoa_ext,Juo3_ext,Juh2o_ext,Jpre_ext,Jtaup_ext) = self.smaccl.run(
                coeffs, tetas_ext, tetav_ext,phis_ext, phiv_ext, uh2o_ext, uo3_ext, 
                taup550_ext, pressure_ext, rtoa_ext, k1p_ext,
                k2p_ext, iaero_ext, 
                XBLOCK=self.XBLOCK, XGRID=self.XGRID, NBLOOP=self.NBLOOP)

        # Output arrays reshaping
        rsurf_ext = np.reshape(rsurf_ext,(Naero, NB,GSIZEXT), order='C')
        Jrtoa_ext = np.reshape(Jrtoa_ext,(NB,GSIZEXT), order='C')
        Juo3_ext  = np.reshape(Juo3_ext, (NB,GSIZEXT), order='C')
        Juh2o_ext = np.reshape(Juh2o_ext,(NB,GSIZEXT), order='C')
        Jpre_ext  = np.reshape(Jpre_ext, (NB,GSIZEXT), order='C')
        Jtaup_ext = np.reshape(Jtaup_ext,(NB,GSIZEXT), order='C')

#        rsurf  = np.zeros((NB,SIZE1,SIZE2, Naero), dtype=np.float32)
#        Drsurf = np.zeros((NB,SIZE1,SIZE2), dtype=np.float32)
#        inter  = np.zeros((SIZE1,SIZE2), dtype=np.float32) + np.nan
#        inter_drtoa  = np.zeros((SIZE1,SIZE2), dtype=np.float32) + np.nan
#        inter_dtaup  = np.zeros((SIZE1,SIZE2), dtype=np.float32) + np.nan
        rsurf  = np.zeros((NB,GSIZE, Naero), dtype=np.float32)
        Drsurf = np.zeros((NB,GSIZE), dtype=np.float32)
        inter  = np.zeros((GSIZE), dtype=np.float32) + np.nan
        inter_drtoa  = np.zeros((GSIZE), dtype=np.float32) + np.nan
        inter_dtaup  = np.zeros((GSIZE), dtype=np.float32) + np.nan
        stock  = np.zeros((GSIZE), dtype=np.float32)

#        if self.breakpoint:
#        Jrtoa  = np.zeros((NB,SIZE1,SIZE2), dtype=np.float32)
#        Juo3   = np.zeros((NB,SIZE1,SIZE2), dtype=np.float32)
#        Juh2o  = np.zeros((NB,SIZE1,SIZE2), dtype=np.float32)
#        Jpre   = np.zeros((NB,SIZE1,SIZE2), dtype=np.float32)
#        Jtaup  = np.zeros((NB,SIZE1,SIZE2), dtype=np.float32)
#        Drtoa  = np.zeros((NB,SIZE1,SIZE2), dtype=np.float32)
#        Duo3   = np.zeros((NB,SIZE1,SIZE2), dtype=np.float32)
#        Duh2o  = np.zeros((NB,SIZE1,SIZE2), dtype=np.float32)
#        Dpre   = np.zeros((NB,SIZE1,SIZE2), dtype=np.float32)
#        Dtaup  = np.zeros((NB,SIZE1,SIZE2), dtype=np.float32)
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
#            Iuo3   = np.zeros((SIZE1,SIZE2), dtype=np.float32) + np.nan
#            Iuh2o  = np.zeros((SIZE1,SIZE2), dtype=np.float32) + np.nan
#            Ipre   = np.zeros((SIZE1,SIZE2), dtype=np.float32) + np.nan
#            Itaup  = np.zeros((SIZE1,SIZE2), dtype=np.float32) + np.nan
#            Ialt   = np.zeros((SIZE1,SIZE2), dtype=np.float32) + np.nan
#            Iaero  = np.zeros((SIZE1,SIZE2), dtype='int32')
            Iuo3   = np.zeros((GSIZE), dtype=np.float32) + np.nan
            Iuh2o  = np.zeros((GSIZE), dtype=np.float32) + np.nan
            Ipre   = np.zeros((GSIZE), dtype=np.float32) + np.nan
            Itaup  = np.zeros((GSIZE), dtype=np.float32) + np.nan
            Ialt   = np.zeros((GSIZE), dtype=np.float32) + np.nan
            Iaero  = np.zeros((GSIZE), dtype='int32')
#            Iuo3[good]   = uo3
#            Iuh2o[good]  = uh2o
#            Ipre[good]   = pressure
#            Itaup[good]  = taup550
#            Ialt[good]   = alt
#            Iaero[good]  = iaero
            Iuo3   = uo3
            Iuh2o  = uh2o
            Ipre   = pressure
            Itaup  = taup550
            Ialt   = alt
            Iaero  = iaero
            ancillary    = [Iuo3,Iuh2o,Itaup,Iaero,Ipre,Ialt]

        for i in range(NB):
            for j in range(Naero):
                inter  = rsurf_ext[j,i,:GSIZE]
#                rsurf[i,:,:,j] = inter[:,:]
                rsurf[i,:,j] = inter[:]

            inter  = abs(Jrtoa_ext[i,:GSIZE]  * rtoa_err[i,:]               ) # it is 0 because no input error
#            if self.breakpoint: 
            inter_drtoa = abs(Jrtoa_ext[i,:GSIZE])
#            Drtoa[i,:,:] = inter_drtoa
            Drtoa[i,:] = inter_drtoa
            stock        = inter**2
            inter  = abs(Jtaup_ext[i,:GSIZE] * (Etaup + ERtaup * taup550  ))
#            if self.breakpoint: 
            inter_dtaup = abs(Jtaup_ext[i,:GSIZE])
#            Dtaup[i,:,:] = inter_dtaup
            Dtaup[i,:] = inter_dtaup
            stock       += inter**2
            inter  = abs(Juo3_ext[i,:GSIZE]  * (Euo3  + ERuo3  * uo3      ))
            #if self.breakpoint: 
#            Duo3[i,:,:]  = inter
            Duo3[i,:]  = inter
            stock       += inter**2
            inter  = abs(Juh2o_ext[i,:GSIZE] * (Euh2o + ERuh2o * uh2o     ))
            #if self.breakpoint: 
#            Duh2o[i,:,:] = inter
            Duh2o[i,:] = inter
            stock       += inter**2
            inter  = abs(Jpre_ext[i,:GSIZE] *  pressure_err)
            #if self.breakpoint: 
#            Dpre[i,:,:]  = inter
            Dpre[i,:]  = inter
            stock       += inter**2

            inter  = np.sqrt(stock)
#            Drsurf[i,:,:]= inter
            Drsurf[i,:]= inter

#            if self.breakpoint:
            inter  = Jrtoa_ext[i,:GSIZE]
#            Jrtoa[i,:,:] = inter
            Jrtoa[i,:] = inter
            inter  = Juo3_ext[i,:GSIZE]
#            Juo3[i,:,:]  = inter
            Juo3[i,:]  = inter
            inter  = Juh2o_ext[i,:GSIZE]
#            Juh2o[i,:,:] = inter
            Juh2o[i,:] = inter
            inter  = Jpre_ext[i,:GSIZE]
#            Jpre[i,:,:]  = inter
            Jpre[i,:]  = inter
            inter  = Jtaup_ext[i,:GSIZE]
#            Jtaup[i,:,:] = inter
            Jtaup[i,:] = inter

        return rsurf, Juh2o, Juo3, Jrtoa, Jpre, Drsurf
        ds_out = xa.Dataset(
            {
                'rsurf': (('bands','Y','X','aer_model'), rsurf),
                'Juh2o': (('bands','Y','X'), Juh2o),
                'Juo3': (('bands','Y','X'), Juo3),
                'Jrtoa': (('bands','Y','X'), Jrtoa),
                'Jpre': (('bands','Y','X'), Jpre),
                'Drsurf': (('bands','Y','X'), Drsurf),
            },
            coords={
                'bands': np.arange(1, NB+1),
                'Y': np.arange(SIZE1),
                'X': np.arange(SIZE2),
                'aer_model': np.arange(1, Naero+1),
            }
        )
        return ds_out 