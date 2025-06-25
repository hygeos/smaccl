from smaccl import Smaccl
import numpy as np
from c3s_lib import set_ac_flag, closest_model_vito, pre_brdf, Ps, dPsdz
import math
import xarray as xa

__module__    = "ISmaccl.py"    
__version__   = "1.01.00"

class ISmaccl(object):
    """
    ISmaccl is a wrapper for the Smaccl class that provides an interface
    for initializing and using the Smaccl library.
    """

    def __init__(self, config, breakpoint=False, ancillary=False, *args, **kwargs):
        """
        Initialize the ISmaccl instance with the provided arguments.
        """
        self.smaccl = Smaccl(*args, **kwargs)
        self.config = config

        # SMACG configuration
        self.XBLOCK = 512
        self.XGRID = 512
        self.NBLOOP = 1
        self.breakpoint = breakpoint
        self.ancillary = ancillary

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying Smaccl instance.
        """
        return getattr(self.smaccl, name)
    
    def run(self, l2_data, merra, dem, frac_aer_model, coeffs, brdf=None):
        """
        Run the ISmaccl instance with the provided arguments.
        """
        sza = l2_data['SZA'].data
        SIZE1, SIZE2 = sza.shape
        good = np.where(sza < 90)

        tetas = l2_data['SZA'].data[good].astype(np.float32, order='C')
        tetav = l2_data['VZA'].data[good].astype(np.float32, order='C')
        phis = l2_data['SAA'].data[good].astype(np.float32, order='C')
        phiv = l2_data['VAA'].data[good].astype(np.float32, order='C')

        lat = l2_data['lat'].data[good]
        lon = l2_data['lon'].data[good]
        t0 = l2_data['mean-time-dec'].data

        taup550 = merra['TOTEXTTAU'].data[good].astype(np.float32, order='C')
        uh2o = merra['TQV'].data[good].astype(np.float32, order='C')
        uo3 = merra['TO3'].data[good].astype(np.float32, order='C')
        p0 = merra['SLP'].data[good].astype(np.float32, order='C')
        t10m = merra['T10M'].data[good].astype(np.float32, order='C')

        alt = dem['elev'].data[good].astype(np.float32, order='C')
        Dalt = dem['Delev'].data[good].astype(np.float32, order='C')
#        # flag large aot pixels
        flag = (taup550 > self.config['taot'])
        l2_data['ac_process_flag'] = (['y','x'], np.zeros(l2_data['SZA'].data.shape, dtype='ubyte'))
        l2_data['ac_process_flag'].data[good] = flag.astype('u1')
        climato = False
        l2_data['ac_flag'] = (['y','x'], np.ones(l2_data['SZA'].data.shape, dtype='uint32'))
        l2_data['ac_flag'].data[good] = set_ac_flag(taup550, tetas, tetav, climato)

        # aerosol model computation
        xb = []
        xm = []

        match = {'sulf':'SU', 'dust':'DU', 'oc':'OC', 'ssalt':'SS', 'bc':'BC'}
        # OPTIONAEROFIXE
        if 'aero_nmod' in self.config.keys():
            nb_pixel = len(lat)
            iaero = np.zeros(nb_pixel)
            iaero[:] = self.config['aero_nmod']
        else:
            for k,key in enumerate(frac_aer_model.keys()):
                frac = merra[match[key]+'_FRAC'].data[good].astype(np.float32, order='C')
                xb.append(frac_aer_model[key])
                xm.append(frac)
            xb = np.stack(xb, axis=0)
            xm = np.stack(xm, axis=0)
            #iaero = closest_model(xm, xb)
            iaero = closest_model_vito(xm, xb)

        tab_band_internal = list(l2_data['bands'].data)
        NB = len(tab_band_internal)
        GSIZE= good[0].size
        # brdf arrays
        # Test for BRDF input data for correction
        if brdf is None:
            k1p = np.zeros((NB, GSIZE), dtype='float32', order='C')
            k2p = np.zeros((NB, GSIZE), dtype='float32', order='C')
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
        rtoa        = np.zeros((NB, GSIZE), dtype='float32', order='C')
        rtoa_err    = np.zeros((NB, GSIZE), dtype='float32', order='C')
        for iband, band in enumerate(tab_band_internal):
            rtoa[iband,:]     = l2_data[band].data[good]
            band_err = '{}_err'.format(band)
            if band_err in l2_data.variables:
                rtoa_err[iband,:] = l2_data[band_err].data[good] 

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
        iaero_ext    = np.zeros((GSIZEXT), dtype='int32')

        for i in range(NB):
            rtoa_ext[i,:GSIZE]= rtoa[i,:]
            k1p_ext[i,:GSIZE] = k1p[i,:]
            k2p_ext[i,:GSIZE] = k2p[i,:]

        taup550_ext[:GSIZE]  = taup550
        uo3_ext[:GSIZE]      = uo3
        pressure_ext[:GSIZE] = pressure
        uh2o_ext[:GSIZE]     = uh2o
        tetas_ext[:GSIZE]    = tetas
        tetav_ext[:GSIZE]    = tetav
        phis_ext[:GSIZE]     = phis
        phiv_ext[:GSIZE]     = phiv
        iaero_ext[:GSIZE]    = iaero

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
        iaero_ext    = np.reshape(iaero_ext   ,(Z,self.XBLOCK,self.XGRID),    order='C')

        (rsurf_ext,Jrtoa_ext,Juo3_ext,Juh2o_ext,Jpre_ext,Jtaup_ext) = self.smaccl.run(
                coeffs, tetas_ext, tetav_ext,phis_ext, phiv_ext, uh2o_ext, uo3_ext, 
                taup550_ext, pressure_ext, rtoa_ext, k1p_ext,
                k2p_ext, iaero_ext, 
                XBLOCK=self.XBLOCK, XGRID=self.XGRID, NBLOOP=self.NBLOOP)
        
        # Output arrays reshaping
        rsurf_ext = np.reshape(rsurf_ext,(NB,GSIZEXT), order='C')
        Jrtoa_ext = np.reshape(Jrtoa_ext,(NB,GSIZEXT), order='C')
        Juo3_ext  = np.reshape(Juo3_ext, (NB,GSIZEXT), order='C')
        Juh2o_ext = np.reshape(Juh2o_ext,(NB,GSIZEXT), order='C')
        Jpre_ext  = np.reshape(Jpre_ext, (NB,GSIZEXT), order='C')
        Jtaup_ext = np.reshape(Jtaup_ext,(NB,GSIZEXT), order='C')

        rsurf  = np.zeros((NB,SIZE1,SIZE2))
        Drsurf = np.zeros((NB,SIZE1,SIZE2))
        inter  = np.zeros((SIZE1,SIZE2)) + np.nan
        inter_drtoa  = np.zeros((SIZE1,SIZE2)) + np.nan
        inter_dtaup  = np.zeros((SIZE1,SIZE2)) + np.nan
        stock  = np.zeros((GSIZE))

        if self.breakpoint:
            Jrtoa  = np.zeros((NB,SIZE1,SIZE2))
            Juo3   = np.zeros((NB,SIZE1,SIZE2))
            Juh2o  = np.zeros((NB,SIZE1,SIZE2))
            Jpre   = np.zeros((NB,SIZE1,SIZE2))
            Jtaup  = np.zeros((NB,SIZE1,SIZE2))
            Drtoa  = np.zeros((NB,SIZE1,SIZE2))
            Duo3   = np.zeros((NB,SIZE1,SIZE2))
            Duh2o  = np.zeros((NB,SIZE1,SIZE2))
            Dpre   = np.zeros((NB,SIZE1,SIZE2))
            Dtaup  = np.zeros((NB,SIZE1,SIZE2))

        if self.ancillary:
            Iuo3   = np.zeros((SIZE1,SIZE2)) + np.nan
            Iuh2o  = np.zeros((SIZE1,SIZE2)) + np.nan
            Ipre   = np.zeros((SIZE1,SIZE2)) + np.nan
            Itaup  = np.zeros((SIZE1,SIZE2)) + np.nan
            Ialt   = np.zeros((SIZE1,SIZE2)) + np.nan
            Iaero  = np.zeros((SIZE1,SIZE2), dtype='int32')
            Iuo3[good]   = uo3
            Iuh2o[good]  = uh2o
            Ipre[good]   = pressure
            Itaup[good]  = taup550
            Ialt[good]   = alt
            Iaero[good]  = iaero
            ancillary    = [Iuo3,Iuh2o,Itaup,Iaero,Ipre,Ialt]

        for i in range(NB):
            inter[good]  = rsurf_ext[i,:GSIZE]
            rsurf[i,:,:] = inter[:,:]

            inter[good]  = abs(Jrtoa_ext[i,:GSIZE]  * rtoa_err[i,:]               ) # it is 0 because no input error
            if self.breakpoint: 
                inter_drtoa[good] = abs(Jrtoa_ext[i,:GSIZE])
                Drtoa[i,:,:] = inter_drtoa
            stock        = inter[good]**2
            inter[good]  = abs(Jtaup_ext[i,:GSIZE] * (Etaup + ERtaup * taup550  ))
            if self.breakpoint: 
                inter_dtaup[good] = abs(Jtaup_ext[i,:GSIZE])
                Dtaup[i,:,:] = inter_dtaup
            stock       += inter[good]**2
            inter[good]  = abs(Juo3_ext[i,:GSIZE]  * (Euo3  + ERuo3  * uo3      ))
            if self.breakpoint: Duo3[i,:,:]  = inter
            stock       += inter[good]**2
            inter[good]  = abs(Juh2o_ext[i,:GSIZE] * (Euh2o + ERuh2o * uh2o     ))
            if self.breakpoint: Duh2o[i,:,:] = inter
            stock       += inter[good]**2
            inter[good]  = abs(Jpre_ext[i,:GSIZE] *  pressure_err)
            if self.breakpoint: Dpre[i,:,:]  = inter
            stock       += inter[good]**2

            inter[good]  = np.sqrt(stock)
            Drsurf[i,:,:]= inter

            if self.breakpoint:
                inter[good]  = Jrtoa_ext[i,:GSIZE]
                Jrtoa[i,:,:] = inter
                inter[good]  = Juo3_ext[i,:GSIZE]
                Juo3[i,:,:]  = inter
                inter[good]  = Juh2o_ext[i,:GSIZE]
                Juh2o[i,:,:] = inter
                inter[good]  = Jpre_ext[i,:GSIZE]
                Jpre[i,:,:]  = inter
                inter[good]  = Jtaup_ext[i,:GSIZE]
                Jtaup[i,:,:] = inter

        toc_data = xa.Dataset({
            'rsurf': (['bands', 'y', 'x'], rsurf),
            'Drsurf': (['bands', 'y', 'x'], Drsurf)
        })
        if self.breakpoint:
            toc_data['Jrtoa'] = [['bands', 'y', 'x'], Jrtoa]
            toc_data['Juo3'] = [['bands', 'y', 'x'], Juo3]
            toc_data['Juh2o'] = [['bands', 'y', 'x'], Juh2o]
            toc_data['Jpre'] = [['bands', 'y', 'x'], Jpre]
            toc_data['Jtaup'] = [['bands', 'y', 'x'], Jtaup]
        if self.ancillary:
            toc_data['Iuo3'] = (['y', 'x'], Iuo3)
            toc_data['Iuh2o'] = (['y', 'x'], Iuh2o)
            toc_data['Itaup'] = (['y', 'x'], Itaup)
            toc_data['Ipre'] = (['y', 'x'], Ipre)
            toc_data['Ialt'] = (['y', 'x'], Ialt)
            toc_data['Iaero'] = (['y', 'x'], Iaero)
            toc_data['ancillary'] = ancillary

        return toc_data