import numpy as np
from luts.luts import read_mlut, MLUT, Idx
from smaccl import Smaccl, Ps, dPsdz, get_smac_coeffs
import xarray
from glob import glob
import math
import h5py
from os.path import basename, exists, dirname
from os import system, walk
from netCDF4 import Dataset
from scipy.interpolate import RectBivariateSpline
from datetime import datetime
import time
import configparser
from sys import argv
from matplotlib.pyplot import imshow, show
from read_s3a_slstr import load

def pre_merra2(faero, fptwo):
    # Read MERRA2 ancillary data files and store all information into a MLUT object for further use
    # (mainly for spatial and temporal interpolation)
    merra = xarray.open_dataset(faero)
    merra_lut = MLUT()
    # Add the good axis
    merra_lut.add_axis('time', date_to_float(merra.time.data)) # float array of delta time in ns from epoch time
    merra_lut.add_axis('lat',  merra.lat.data)
    merra_lut.add_axis('lon',  merra.lon.data)
    merra_lut.add_dataset('TOTEXTTAU', merra['TOTEXTTAU'].data, axnames=['time','lat','lon'])

    merra = xarray.open_dataset(fptwo)
    merra_lut.add_dataset('TO3',  merra['TO3'].data,  axnames=['time','lat','lon'])
    merra_lut.add_dataset('SLP',  merra['SLP'].data,  axnames=['time','lat','lon'])
    merra_lut.add_dataset('T10M', merra['T10M'].data, axnames=['time','lat','lon'])
    merra_lut.add_dataset('TQV',  merra['TQV'].data,  axnames=['time','lat','lon'])

    del merra

    return merra_lut

def date_to_float(d, epoch=np.datetime64('1980-01-01T00:00:00.000000000')):
    '''
        transform the date into a duration in minutes since epoch
            '''
    return (d - epoch).astype(np.float64)/1.0e9/60.

def create_nc(filename, gl_size, attrs, version):
    print("save netCDF : {}".format(filename))
    out = Dataset(filename, 'w', format='NETCDF4')

    for att, value in attrs:
        out.setncattr(att, value)

    out.date_created = str(datetime.now())
    out.production_centre = 'vito'
    out.version = version

    width = gl_size[0]
    height = gl_size[1]
    h = out.createDimension('height', height)
    w = out.createDimension('width', width)

    return out

def save_nc(out, data, rsurf, Drsurf, version, dataset_names, bandidx, bandsize): #, Irtoa0):#, Irtoa1, Irtoa2, Ipre, Iuo3, Iuh2o, Itaup, Ialt):
    size = data['lat'].data.shape

    ymin = bandidx*bandsize
    ymax = ymin + bandsize

    if bandidx==0: sds = out.createVariable('S4_an', 'f', ('height','width'), complevel=9)
    else: sds = out['S4_an']
    sds[ymin:ymax,:] = data['S4_radiance_an'].data[:,:]

    for idx in range(rsurf.shape[0]):
        band = dataset_names[idx].replace('_radiance','')
        if bandidx==0 : sds = out.createVariable('TOC_{}'.format(band), 'f', ('height','width'), complevel=9)
        else: sds = out['TOC_{}'.format(band)]
        sds[ymin:ymax, :] = rsurf[idx]
        sds.Long_name = 'Top of Canopy Reflectance'
        sds.Unit = 'None'
        band = 'TOC_{} error'.format(band)
        if bandidx==0: sds = out.createVariable(band, 'f', ('height','width'), complevel=9)
        else: sds = out[band]
        sds[ymin:ymax,:] = Drsurf[idx]
        sds.Long_name = 'Uncertainty Top of Canopy Reflectance'
        sds.Unit = 'None'
    if bandidx==0: sds = out.createVariable('Lat', 'f', ('height','width'), complevel=9)
    else: sds = out['Lat']
    sds[ymin:ymax,:] = data['lat'].data
    sds.Unit = 'Degree'
    if bandidx==0: sds = out.createVariable('Lon', 'f', ('height', 'width'), complevel=9)
    else: sds = out['Lon']
    sds[ymin:ymax,:] = data['lon'].data
    sds.Unit = 'Degree'
    
    if bandidx == 0: sds = out.createVariable('SZA', 'f', ('height', 'width'), complevel=9)
    else: sds = out['SZA']
    sds[ymin:ymax,:] = data['SZA'].data
    sds.Unit = 'Degree'
    if bandidx==0 : sds = out.createVariable('SAA', 'f', ('height', 'width'), complevel=9)
    else: sds = out['SAA']
    sds[ymin:ymax,:] = data['SAA'].data
    sds.Unit = 'Degree'
    if bandidx==0: sds = out.createVariable('VZA', 'f', ('height', 'width'), complevel=9)
    else: sds = out['VZA']
    sds[ymin:ymax,:] = data['VZA'].data
    sds.Unit = 'Degree'
    if bandidx==0: sds = out.createVariable('VAA', 'f', ('height', 'width'), complevel=9)
    else: sds = out['VAA']
    sds[ymin:ymax,:] = data['VAA'].data
    sds.Unit = 'Degree'

    if bandidx==0: sds = out.createVariable('cloud_an', 'u2', ('height', 'width'), complevel=9)
    else: sds = out['cloud_an']
    sds[ymin:ymax,:] = data['cloud_an'].data
    if bandidx==0: sds = out.createVariable('quality_flags', 'u4', ('height', 'width'), complevel=9)
    else: sds = out['quality_flags']
    sds[ymin:ymax,:] = data['quality_flags'].data
    if bandidx==0: sds = out.createVariable('pixel_classif_flags', 'u2', ('height', 'width'), complevel=9)
    else: sds = out['pixel_classif_flags']
    sds[ymin:ymax,:] = data['pixel_classif_flags'].data

        
def process(config, dem_lut, S):

    fname = config['input']
    fileout = config['output']

    # SMACG configuration
    XBLOCK = 512
    XGRID = 512
    YGRID = 1
    YBLOCK = 1
    NBLOOP = 1

    aer_coef = 'CONT'
    dir_coef_name = './'

    version = '1.0'

    k_uh2o = config['k_uh2o']
    k_uo3 = config['k_uo3']
    k_p0 = config['k_p0']
#
    Etoa = config['etoa']
    ERtoa = config['ertoa'] 
    Etaup = config['etaup']
    ERtaup = config['ertaup']
    Euo3 = config['euo3']
    ERuo3 = config['eruo3']
    Euh2o = config['euh2o']
    ERuh2o = config['eruh2o']
    Epre = config['epre']
    ERpre = config['erpre']

    smaccoef = {'olci':config['smaccoef_olci'], 'slstr':config['smaccoef_slstr']}
    nbchunk = 4
    bandsize = 3360//nbchunk
    data, SIZE1, SIZE2, tab_band_internal, _, _ , coeffs, gl_size = load(fname, smaccoef, 0, bandsize)
    out = create_nc(fileout, gl_size, data.attrs.items(), version)

    for bandidx in range(nbchunk):
        data, SIZE1, SIZE2, tab_band_internal, _, _, coeffs, gl_size = load(fname, smaccoef, bandidx, bandsize)

        if data is None:
            print("l'image n'a pas d'attributs")
            return

        year = str(data['mean-time'].values)[:4]
        month = str(data['mean-time'].values)[5:7]
        day = str(data['mean-time'].values)[8:10]

        # path to input ancillary MERRA 2 data
        # 1) aerosols
        merra_aerosol = config['merraaero']
        # 2) PTWO, Pressure, Temperature, Water vapour ,Ozone
        merra_ptwo = config['merraptwo']

        merra_lut = pre_merra2(merra_aerosol, merra_ptwo)

        # files containg SMAC coefficients

        # masking cloudy & out of orbit pixels and SZA above 90°
        SM   = data['clm'].data
        SZA  = data['SZA'].data
        good = np.where((SM == 0) & (SZA < 90))
        print(good)
        data['S4_radiance_an'].data[~((SM==0)&(SZA<90))] = np.NaN
        if len(good[0]) == 0:
            print('image nuageuse')
            return
        GSIZE= good[0].size
        NB = len(tab_band_internal)

        # start with tie points
        tetas       = data['SZA'].data[good].astype(np.float32, order='C')
        tetav       = data['VZA'].data[good].astype(np.float32, order='C')
        phis        = data['SAA'].data[good].astype(np.float32, order='C')
        phiv        = data['VAA'].data[good].astype(np.float32, order='C')

        lat         = data['lat'].data[good]
        lon         = data['lon'].data[good]
        t0          = data['mean-time-dec'].data

        #interpolate merra 2 data and dem to the image location and time
        taup550 = merra_lut['TOTEXTTAU'][Idx(t0,  round=False, fill_value='extrema'), Idx(lat, round=False, fill_value='extrema'), Idx(lon, round=False, fill_value='extrema')].astype(np.float32, order='C') 
        uh2o = merra_lut['TQV'][Idx(t0,  round=False, fill_value='extrema'), Idx(lat, round=False, fill_value='extrema'), Idx(lon, round=False, fill_value='extrema')].astype(np.float32, order='C') 
        uo3 = merra_lut['TO3'][Idx(t0,  round=False, fill_value='extrema'), Idx(lat, round=False, fill_value='extrema'), Idx(lon, round=False, fill_value='extrema')].astype(np.float32, order='C') 
        p0 = merra_lut['SLP'][Idx(t0,  round=False, fill_value='extrema'), Idx(lat, round=False, fill_value='extrema'), Idx(lon, round=False, fill_value='extrema')].astype(np.float32, order='C') 
        t10m = merra_lut['T10M'][Idx(t0,  round=False, fill_value='extrema'), Idx(lat, round=False, fill_value='extrema'), Idx(lon, round=False, fill_value='extrema')].astype(np.float32, order='C')
        #interpolate DEM and uncertainty to the image location and time
        alt = np.array(dem_lut['elev'][Idx(lat, round=False, fill_value='extrema'), Idx(lon, round=False, fill_value='extrema')]).astype(np.float32, order='C') 
        Dalt = np.array(dem_lut['Delev'][Idx(lat, round=False, fill_value='extrema'), Idx(lon, round=False, fill_value='extrema')]).astype(np.float32, order='C')

        del lat
        del lon
        del t0
        del merra_lut
        # pressure correction for surface altitude and transformation from Pa to hPa
        pressure = Ps(alt, p0*k_p0, t10m)
        # quadratic mean of error due to met fields (Epre) and error due to altitude (Dalt)
        pressure_err = np.sqrt((dPsdz(alt, p0*k_p0, t10m) * Dalt)**2 + Epre**2)/2.
        # cndsonversion from kg.m-2 to g.cm-2
        uh2o *= k_uh2o
        # conversion from Dobson to cm.atm 
        uo3  *= k_uo3

        # prepare radiometry array
        rtoa        = np.zeros((NB, GSIZE), dtype='float32', order='C')
        rtoa_err    = np.zeros((NB, GSIZE), dtype='float32', order='C')
        for iband, band in enumerate(tab_band_internal):
            rtoa[iband,:]     = data[band].data[good]
            #rtoa_err = 0 because no input
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
        for i in range(NB):
            rtoa_ext[i,:GSIZE] = rtoa[i,:]
        del rtoa

        taup550_ext[:GSIZE]  = taup550
        uo3_ext[:GSIZE]      = uo3
        pressure_ext[:GSIZE] = pressure
        uh2o_ext[:GSIZE]     = uh2o
        tetas_ext[:GSIZE]    = tetas
        tetav_ext[:GSIZE]    = tetav
        phis_ext[:GSIZE]     = phis
        phiv_ext[:GSIZE]     = phiv
#        del pressure
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

        (rsurf_ext,Jrtoa_ext,Juo3_ext,Juh2o_ext,Jpre_ext,Jtaup_ext) = S.run(coeffs, tetas_ext, tetav_ext,phis_ext, phiv_ext, uh2o_ext, uo3_ext, taup550_ext, pressure_ext, rtoa_ext,XBLOCK=XBLOCK, XGRID=XGRID, NBLOOP=NBLOOP)

        # Output arrays reshaping \n",
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

        BREAKPOINT = False
        INPUT = False

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

        if INPUT:
            Irtoa  = np.zeros((NB,SIZE1,SIZE2))
            Iuo3   = np.zeros((SIZE1,SIZE2)) + np.nan
            Iuh2o  = np.zeros((SIZE1,SIZE2)) + np.nan
            Ipre   = np.zeros((SIZE1,SIZE2)) + np.nan
            Itaup  = np.zeros((SIZE1,SIZE2)) + np.nan
            Ialt   = np.zeros((SIZE1,SIZE2)) + np.nan
            Iuo3[good]   = uo3
            Iuh2o[good]  = uh2o
            Ipre[good]   = pressure
            Itaup[good]  = taup550
            Ialt[good]   = alt

        for i in range(NB):
            inter[good]  = rsurf_ext[i,:GSIZE]
            rsurf[i,:,:] = inter

            if INPUT:
                inter[good] = rtoa[i,:]
                Irtoa[i,:,:]= inter

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

        save_nc(out, data, rsurf, Drsurf, version, tab_band_internal, bandidx, bandsize) #, Irtoa[0])

    out.close()

def readConfig(configfile):
    cf = configparser.ConfigParser()
    cf.read(configfile)
    config = {}
    for k,v  in cf.items('Coefficients'):
       config[k] = float(v) 
    for k,v in cf.items('Paths'):
        config[k] = v

    return config

def main(configfile):
    config = readConfig(configfile)

    if not(exists(config['input'])):
        print('file "{}" does not exist'.format(config['input']))
        exit(0)

    dem_lut = read_mlut(config['dem'])

    S = Smaccl('CPU')

    process(config, dem_lut, S)

if __name__=='__main__':
    
    fileconfig = argv[1] #'config.cfg'
    main(fileconfig)
