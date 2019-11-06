# encoding: utf-8

import numpy as np
from luts.luts import read_mlut, Idx, MLUT
from smaccl import Smaccl
from srtm import SRTM3
import math
import configparser
from sys import argv
from os.path import exists
from c3s_io import load_olci_slstr, load_msi, load_oli, save_nc, create_nc, load_testcase_vito
from c3s_lib import Ps, dPsdz, pre_merra2, pre_aer_models, closest_model, load_cams, set_ac_flag


def process(config, dem, S, BREAKPOINT=False, ANCILLARY=False):

    fname = config['input']
    fileout = config['output']

    # SMACG configuration
    XBLOCK = 512
    XGRID = 512
    NBLOOP = 1

    version = '1.0'

    k_uh2o = config['k_uh2o']
    k_uo3 = config['k_uo3']
    k_p0 = config['k_p0']
    Etaup = config['etaup']
    ERtaup = config['ertaup']
    Euo3 = config['euo3']
    ERuo3 = config['eruo3']
    Euh2o = config['euh2o']
    ERuh2o = config['eruh2o']
    Epre = config['epre']
    nbchunk = config['nbchunk']
    imsize  = config['imsize']
    platform= config['sensor'].split(sep='_')[0]
    sensors = config['sensor'].split(sep='_')[1:]
    smaccoef = {}
    if 'resolution' in config.keys():
        resolution = config['resolution']
    else: resolution = '60'

    for s in sensors: smaccoef[s.lower()]=config['smaccoef_dir']+platform+'_'+s+'_smac_coeffs.npy'
    #
    if 'S3' in platform : 
        data, _,  _, _, _, _ , _, gl_size = load_olci_slstr(fname, smaccoef, 0,  1, platform=platform)
    elif 'LANDSAT' in platform : 
        data, SIZE1, SIZE2, _, _, _ , _, gl_size = load_oli(fname, smaccoef, 0, -1, platform=platform)
    elif 'S2' in platform : 
        data, SIZE1, SIZE2, _, _, _ , _, gl_size = load_msi(fname, smaccoef, 0, -1, 
                                                            platform=platform, resolution=resolution)
    elif 'VITO' in platform:
        data, SIZE1, SIZE2, tab_band_internal, coeffs, gl_size = load_testcase_vito(fname, config['smaccoef_dir'], sensors[0])

    if data is None:
        return
    out = create_nc(fileout, gl_size, data.attrs.items(), version)
    if imsize < 0 : imsize = gl_size[0]
    chunksize = imsize//nbchunk
    if imsize%nbchunk!=0:
        chunksize +=1

    # TODO: test sur l'existance de données auxilliaires sinon utilisation 
    # de la climato et passage du 2eme bit de ac_process_flag a 1.

    if 'cams' in config.keys():
        print("ancillary CAMS")
        merra_lut = load_cams(config['cams'])
    else:
        print('ancillary MERRA 2')
        # path to input ancillary MERRA 2 data
        # 1) aerosols
        merra_aerosol = config['merraaero']
        # 2) PTWO, Pressure, Temperature, Water vapour ,Ozone
        merra_ptwo = config['merraptwo']
        # 3) Aerosols components fraction
        faer       = config['faer']

    for chunkidx in range(nbchunk):
        ancillary = None
        match = {'sulf':'SU', 'dust':'DU', 'oc':'OC', 'ssalt':'SS', 'bc':'BC'}
        if 'S3' in platform : 
            data, SIZE1, SIZE2, tab_band_internal, _, _, coeffs, gl_size = load_olci_slstr(fname, smaccoef, chunkidx, chunksize, platform=platform)
        elif 'LANDSAT' in platform : 
            data, SIZE1, SIZE2, tab_band_internal, _, _, coeffs, gl_size = load_oli(fname, smaccoef, chunkidx, chunksize, platform=platform)
        elif 'S2' in platform : 
            data, SIZE1, SIZE2, tab_band_internal, _, _, coeffs, gl_size = load_msi(fname, smaccoef, chunkidx, chunksize, 
                                                                                    platform=platform, resolution=resolution)

        if data is None:
            print("Image has empty")
            good = [[],[]]
        else:
            data['ac_process_flag'] = (['y','x'], np.zeros(data['SZA'].data.shape, dtype='ubyte'))
            data['ac_flag'] = (['y','x'], np.ones(data['SZA'].data.shape, dtype='uint32'))
            # masking cloudy & out of orbit pixels and SZA above 90°
            SM   = data['clm'].data
            SZA  = data['SZA'].data
            good = np.where((SM == 0) & (SZA < 90))
            NB = len(tab_band_internal)

        if len(good[0]) == 0:
            print('totally cloudy chunk #{}'.format(chunkidx))
            rsurf  = np.zeros((NB,SIZE1,SIZE2)) + np.NaN
            Drsurf = np.zeros((NB,SIZE1,SIZE2)) + np.NaN

        else:
            if not('cams' in config.keys()):
                merra_lut = pre_merra2(merra_aerosol, merra_ptwo)
            frac_aer_model = pre_aer_models(faer)

            # start with tie points
            tetas       = data['SZA'].data[good].astype(np.float32, order='C')
            tetav       = data['VZA'].data[good].astype(np.float32, order='C')
            phis        = data['SAA'].data[good].astype(np.float32, order='C')
            phiv        = data['VAA'].data[good].astype(np.float32, order='C')

            GSIZE= good[0].size
            print('cloud free pixels in chunk #{} : {}'.format(chunkidx,GSIZE))

            lat         = data['lat'].data[good]
            lon         = data['lon'].data[good]
            t0          = data['mean-time-dec'].data

            #interpolate merra 2 data and dem to the image location and time
            taup550 = merra_lut['TOTEXTTAU'][Idx(t0,  round=False, fill_value='extrema'),
                                             Idx(lat, round=False, fill_value='extrema'), 
                                             Idx(lon, round=False, fill_value='extrema')].astype(np.float32, order='C') 
            uh2o    = merra_lut['TQV'][Idx(t0,  round=False, fill_value='extrema'),
                                       Idx(lat, round=False, fill_value='extrema'), 
                                       Idx(lon, round=False, fill_value='extrema')].astype(np.float32, order='C') 
            uo3     = merra_lut['TO3'][Idx(t0,  round=False, fill_value='extrema'), 
                                       Idx(lat, round=False, fill_value='extrema'), 
                                       Idx(lon, round=False, fill_value='extrema')].astype(np.float32, order='C') 
            p0      = merra_lut['SLP'][Idx(t0,  round=False, fill_value='extrema'),
                                       Idx(lat, round=False, fill_value='extrema'), 
                                       Idx(lon, round=False, fill_value='extrema')].astype(np.float32, order='C') 
            t10m    = merra_lut['T10M'][Idx(t0,  round=False, fill_value='extrema'), 
                                        Idx(lat, round=False, fill_value='extrema'), 
                                        Idx(lon, round=False, fill_value='extrema')].astype(np.float32, order='C')
            #interpolate DEM and uncertainty to the image location and time
            if isinstance(dem, MLUT):
                alt     = np.array(dem['elev'][Idx(lat, round=False, fill_value='extrema'), 
                                               Idx(lon, round=False, fill_value='extrema')]).astype(np.float32, order='C') 
                Dalt    = np.array(dem['Delev'][Idx(lat, round=False, fill_value='extrema'), 
                                                Idx(lon, round=False, fill_value='extrema')]).astype(np.float32, order='C')
            elif isinstance(dem, SRTM3) :
                alt     = dem.get(lat, lon).astype(np.float32, order='C')
                Dalt    = np.zeros_like(alt)

            # flag large aot pixels
            flag  = (taup550 > config['taot'])
            data['ac_process_flag'].data[good] |= flag.astype('u1')
            climato = False
            data['ac_flag'].data[good] = set_ac_flag(taup550, tetas, tetav, climato)

            # aerosol model computation
            xb = []
            xm = []
            for k,key in enumerate(frac_aer_model.keys()):
                frac = merra_lut[match[key]+'_FRAC'][Idx(t0,  round=False, fill_value='extrema'),
                                             Idx(lat, round=False, fill_value='extrema'), 
                                             Idx(lon, round=False, fill_value='extrema')].astype(np.float32, order='C')              
                xb.append(frac_aer_model[key])
                xm.append(frac)
            xb = np.stack(xb, axis=0)
            xm = np.stack(xm, axis=0)
            xb = xb[:, :         ,np.newaxis]
            xm = xm[:, np.newaxis,         :]
            iaero = closest_model(xm, xb)
    
            del lat
            del lon
            del t0
            del merra_lut

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
                rtoa[iband,:]     = data[band].data[good]
                band_err = '{}_err'.format(band)
                rtoa_err[iband,:] = data[band_err].data[good] 
                del data[band]

            Z = int(math.ceil(float(GSIZE)/float(XBLOCK*XGRID)))
            GSIZEXT = Z * XBLOCK * XGRID

            # the \"ext\" suffix is for extended arrays, larger than the good pixels size, it is completed by NaN's\n",
            rtoa_ext     = np.zeros((NB, GSIZEXT), dtype='float32') + np.NaN
            taup550_ext  = np.zeros((GSIZEXT), dtype='float32') + np.NaN                                                                                                               
            uo3_ext      = np.zeros((GSIZEXT), dtype='float32') + np.NaN
            pressure_ext = np.zeros((GSIZEXT), dtype='float32') + np.NaN
            uh2o_ext     = np.zeros((GSIZEXT), dtype='float32') + np.NaN
            tetas_ext    = np.zeros((GSIZEXT), dtype='float32') + np.NaN
            tetav_ext    = np.zeros((GSIZEXT), dtype='float32') + np.NaN
            phis_ext     = np.zeros((GSIZEXT), dtype='float32') + np.NaN
            phiv_ext     = np.zeros((GSIZEXT), dtype='float32') + np.NaN
            iaero_ext    = np.zeros((GSIZEXT), dtype='int32')

            for i in range(NB):
                rtoa_ext[i,:GSIZE] = rtoa[i,:]
            #del rtoa

            taup550_ext[:GSIZE]  = taup550
            uo3_ext[:GSIZE]      = uo3
            pressure_ext[:GSIZE] = pressure
            uh2o_ext[:GSIZE]     = uh2o
            tetas_ext[:GSIZE]    = tetas
            tetav_ext[:GSIZE]    = tetav
            phis_ext[:GSIZE]     = phis
            phiv_ext[:GSIZE]     = phiv
            iaero_ext[:GSIZE]    = iaero

            #del pressure
            del tetas
            del tetav
            del phis
            del phiv

            # Input arrays reshaping
            rtoa_ext     = np.reshape(rtoa_ext,    (NB,Z,XBLOCK,XGRID), order='C')
            tetas_ext    = np.reshape(tetas_ext,   (Z,XBLOCK,XGRID),    order='C')
            tetav_ext    = np.reshape(tetav_ext,   (Z,XBLOCK,XGRID),    order='C')
            phis_ext     = np.reshape(phis_ext,    (Z,XBLOCK,XGRID),    order='C')
            phiv_ext     = np.reshape(phiv_ext,    (Z,XBLOCK,XGRID),    order='C')
            uh2o_ext     = np.reshape(uh2o_ext,    (Z,XBLOCK,XGRID),    order='C')
            uo3_ext      = np.reshape(uo3_ext,     (Z,XBLOCK,XGRID),    order='C')
            taup550_ext  = np.reshape(taup550_ext, (Z,XBLOCK,XGRID),    order='C')
            pressure_ext = np.reshape(pressure_ext,(Z,XBLOCK,XGRID),    order='C')
            iaero_ext    = np.reshape(iaero_ext   ,(Z,XBLOCK,XGRID),    order='C')

            (rsurf_ext,Jrtoa_ext,Juo3_ext,Juh2o_ext,Jpre_ext,Jtaup_ext) = S.run(
                    coeffs, tetas_ext, tetav_ext,phis_ext, phiv_ext, uh2o_ext, uo3_ext, 
                    taup550_ext, pressure_ext, rtoa_ext, iaero_ext, 
                    XBLOCK=XBLOCK, XGRID=XGRID, NBLOOP=NBLOOP)

            if config['sensor']=='VITO_PROBAV':
                tetav       = data['VZA_SWIR'].data[good].astype(np.float32, order='C')
                tetav_ext    = np.zeros((GSIZEXT), dtype='float32') + np.NaN
                tetav_ext[:GSIZE]    = tetav
                tetav_ext    = np.reshape(tetav_ext,   (Z,XBLOCK,XGRID),    order='C')
                phiv        = data['VAA_SWIR'].data[good].astype(np.float32, order='C')
                phiv_ext     = np.zeros((GSIZEXT), dtype='float32') + np.NaN
                phiv_ext[:GSIZE]     = phiv
                phiv_ext     = np.reshape(phiv_ext,    (Z,XBLOCK,XGRID),    order='C')
                rtoa_swir_ext     = np.zeros((1, GSIZEXT), dtype='float32') + np.NaN
                rtoa_swir_ext[0,:GSIZE] = rtoa[-1,:]
                rtoa_swir_ext     = np.reshape(rtoa_swir_ext,    (1,Z,XBLOCK,XGRID), order='C')
                coeffs_swir = coeffs[-1]
                coeffs_swir = np.reshape(coeffs_swir, (1, coeffs[-1].shape[0]))

                (rsurf_swir_ext,Jrtoa_swir_ext,Juo3_swir_ext,Juh2o_swir_ext,Jpre_swir_ext,Jtaup_swir_ext) = S.run(
                    coeffs_swir, tetas_ext, tetav_ext,phis_ext, phiv_ext, uh2o_ext, uo3_ext, 
                    taup550_ext, pressure_ext, rtoa_swir_ext, iaero_ext, 
                    XBLOCK=XBLOCK, XGRID=XGRID, NBLOOP=NBLOOP)

                rsurf_ext[-1] = rsurf_swir_ext
                Jrtoa_ext[-1] = Jrtoa_swir_ext
                Juo3_ext[-1] = Juo3_swir_ext
                Juh2o_ext[-1] = Juh2o_swir_ext
                Jpre_ext[-1] = Jpre_swir_ext
                Jtaup_ext[-1] = Jtaup_swir_ext

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
            stock  = np.zeros((GSIZE))

            if BREAKPOINT:
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


            if ANCILLARY:
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
                rsurf[i,:,:] = inter

                inter[good]  = abs(Jrtoa_ext[i,:GSIZE]  * rtoa_err[i,:]               ) # it is 0 because no input error
                if BREAKPOINT: Drtoa[i,:,:] = inter
                stock        = inter[good]**2
                inter[good]  = abs(Jtaup_ext[i,:GSIZE] * (Etaup + ERtaup * taup550  ))
                if BREAKPOINT: Dtaup[i,:,:] = inter
                stock       += inter[good]**2
                inter[good]  = abs(Juo3_ext[i,:GSIZE]  * (Euo3  + ERuo3  * uo3      ))
                if BREAKPOINT: Duo3[i,:,:]  = inter
                stock       += inter[good]**2
                inter[good]  = abs(Juh2o_ext[i,:GSIZE] * (Euh2o + ERuh2o * uh2o     ))
                if BREAKPOINT: Duh2o[i,:,:] = inter
                stock       += inter[good]**2
                inter[good]  = abs(Jpre_ext[i,:GSIZE] *  pressure_err)
                if BREAKPOINT: Dpre[i,:,:]  = inter
                stock       += inter[good]**2

                inter[good]  = np.sqrt(stock/5.)
                Drsurf[i,:,:]= inter

                if BREAKPOINT:
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

            del inter
            del stock

        save_nc(out, data, rsurf, Drsurf, version, tab_band_internal, chunkidx, chunksize, ancillary=ancillary, sensor=sensors[0])

    out.close()


def readConfig(configfile):
    cf = configparser.ConfigParser()
    cf.read(configfile)
    config = {}
    for k,v  in cf.items('Coefficients'):
       config[k] = float(v) 
    for k,v  in cf.items('Sizes'):
       config[k] = int(v) 
    for k,v in cf.items('Paths'):
        config[k] = v
    for k,v in cf.items('Sensor'):
        config[k] = v

    return config

def main(configfile):
    config = readConfig(configfile)

    if not(exists(config['input'])):
        print('file "{}" does not exist'.format(config['input']))
        exit(0)

    dem = SRTM3(directory=config['dem'], missing=0.0)

    S = Smaccl('CPU')

    process(config, dem, S)

    print('end')

if __name__=='__main__':
    
    fileconfig = argv[1] #'config.cfg'
    main(fileconfig)
