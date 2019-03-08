import numpy as np
from luts import read_mlut, MLUT, Idx
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

    # MLUT object description
#    merra_lut.describe()

    del merra

    return merra_lut

def date_to_float(d, epoch=np.datetime64('1980-01-01T00:00:00.000000000')):
    '''
        transform the date into a duration in minutes since epoch
            '''
    return (d - epoch).astype(np.float64)/1.0e9/60.

def pre_image(fname, aer_coef):

    def VGT_TIE_COMPLETE(dataset, a, size, tie):
        XSIZE, YSIZE = size
        XTIE, YTIE = tie
        ref = dataset[a] * float(dataset[a].attrs['Scale']) + float(dataset[a].attrs['Offset'])
        r1 = np.arange(0, XSIZE, XSIZE/XTIE)
        r2 = np.arange(0, YSIZE, YSIZE/YTIE)
        r11,r22 = np.meshgrid(np.arange(XSIZE), np.arange(YSIZE))
        res = RectBivariateSpline(r1, r2, ref).ev(r11, r22)
        res = np.swapaxes(res, 0, 1)
        res = xarray.DataArray(res ,dims = ('phony_dim_0','phony_dim_0'))

        dataset.drop(a)
        dataset[a] = res

    def get_meantime(dataset):
        time = dataset.time_coverage_start
        dt1  = np.datetime64(str(time[:4]) + '-' + str(time[4:6]) + '-' + str(time[6:8]) + 'T' + str(time[9:11]) + ':' + str(time[11:13]) + ':' + str(time[13:15]))
        time = dataset.time_coverage_end
        dt2  = np.datetime64(str(time[:4]) + '-' + str(time[4:6]) + '-' + str(time[6:8]) + 'T' + str(time[9:11]) + ':' + str(time[11:13]) + ':' + str(time[13:15]))

        return ((dt2-dt1)/2. +dt1)

    def get_meantime_s3a(dateset):
        day = xdataset.start_date[:2]
        year = xdataset.start_date[7:11]
        month = xdataset.start_date[3:6]
        hour = xdataset.start_date[12:14]
        minu = xdataset.start_date[15:17]
        sec = xdataset.start_date[18:20]
        strmonths = np.array(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
        month = np.where(month==strmonths)[0][0]
        dt = np.datetime64(year + '-' + '{:02d}'.format(month) + '-' + day + 'T' + hour + ':' + minu + ':' + sec)
        return dt


    def to_float (dataset, elem):
        res = dataset[elem] = dataset[elem] * float(dataset[elem].attrs['Scale']) + float(dataset[elem].attrs['Offset'])

        return res

    try:
        xdataset = xarray.open_dataset(fname)
    except:
       return None, None, None, None, None
        

    try:
        sensor = xdataset.sensor
    except:
        if 'S3A' in fname.split('/')[-1]:
#        if fname.split('/')[-1][:3] == 'S3A':
            sensor = 'S3A'
        else:
            sensor = 'VGT'

    if sensor=='VEGETATION':
        sensor = 'VGT'

    # sensor switch
#    sensor = 'AVHRR/3' # forcage car l'attribut sensor n'existe pas toujours dans les images testdata
    if sensor == 'AVHRR/3' or sensor == 'AVHRR/2':
       conv = {'ch1':'b1','ch2':'b2','sun_zen':'SZA', 'sun_azi':'SAA','sat_zen':'VZA','sat_azi':'VAA'}
       xdataset.rename(conv, inplace=True)
       tab_band_internal = ['b1','b2']
       smac_coeff_name = ['VIS', 'NIR']

#       if 'avhrr_b3a' in xdataset.data_vars.keys():
       if 'ch3a' in xdataset.data_vars.keys() and exists('./COEFFS/coef_NOAA09_MIR_CONT.dat'):
           tab_band_internal.append('b3a')
           conv = {'ch3a':'b3a'}
           xdataset.rename(conv, inplace=True)
           smac_coeff_name.append('MIR')

       SIZE1, SIZE2 = xdataset[tab_band_internal[0]].shape

       for band in tab_band_internal:
           if not('{}_unc'.format(band) in xdataset.data_vars.keys()):
               void = xarray.DataArray(np.zeros((SIZE1, SIZE2), dtype='float32') + np.NaN, coords=[xdataset.Latitude,xdataset.Longitude], dims=['Latitude','Longitude'])
               xdataset['{}_unc'.format(band)] = void

       new_attrs = {'Scale':0.01, 'Offset':0.0} #
       for band in tab_band_internal:
           xdataset[band] = xdataset[band].assign_attrs(new_attrs)

       try: 
#           platform = xdataset.platform.replace('-','')
            dummy = xdataset.platform.split('-')
            platform = '{}{:02d}'.format(dummy[0], int(dummy[1]))
       except:
           return None, None, None, None, None

       smac_coeff_name = ['coef_{}_{}_{}.dat'.format(platform, x, aer_coef) for x in smac_coeff_name]

       lon,lat=np.meshgrid(xdataset['Longitude'],xdataset['Latitude'])
       xdataset['lat']=(('x', 'y'), lat)
       xdataset['lon']=(('x', 'y'), lon)

       # scale ref
       for band in tab_band_internal:
           to_float(xdataset, band)
                                                       
       # mean decimal time for the scene
       xdataset['mean-time'] = get_meantime(xdataset)
       xdataset['mean-time-dec'] = date_to_float(xdataset['mean-time'].data)

       for band in tab_band_internal:
           to_float(xdataset, band)
           bandunc = '{} uncertainty'.format(band)
           if  bandunc in xdataset.data_vars.keys():
               to_float(xdataset, bandunc)
           else:
               void = xarray.DataArray(np.zeros((SIZE1, SIZE2)) + np.NaN, coords=[xdataset.Latitude, xdataset.Longitude], dims=['Latitude','Longitude'])
               xdataset[bandunc] = void
           xdataset.rename({bandunc:'{}_unc'.format(band)}, inplace=True)
    elif sensor=='VGT':

#        if 'segm_reference' in xdataset.__dict__:
        ref = xdataset.segm_reference
#        else:
#            ref = xdataset.SEGM_REFERENCE

        if ref[:2] == 'V1':
            sensor = 'VGT1'
        else:
            sensor = 'VGT2'

        tab_band_internal = ['B0','B2','B3','MIR']
        smac_coeff_name = ['coef_{}_{}_{}.dat'.format(sensor, str(x), aer_coef) for x in tab_band_internal]
        SIZE1, SIZE2 = xdataset[tab_band_internal[0]].shape
        XTIE, YTIE = xdataset['SZA'].shape

        lon0, lat0, d_lon, d_lat = [float(x) for x in xdataset.map_info.split(',')[3:7]]
        xdataset['Latitude'] = lat0 - np.arange(SIZE1)*d_lat
        xdataset['Longitude'] = lon0 + np.arange(SIZE2)*d_lon

        date_vgt = xdataset.segm_first_date
        time_vgt = xdataset.segm_first_time
        first_date = '{}T{}'.format(date_vgt, time_vgt)

        date_vgt = xdataset.segm_last_date
        time_vgt = xdataset.segm_last_time
        last_date = '{}T{}'.format(date_vgt, time_vgt)
        new_attrs = {'time_coverage_start':first_date, 'time_coverage_end':last_date}
        xdataset = xdataset.assign_attrs(new_attrs)

        clm = np.ones((SIZE1, SIZE2), dtype='int8')
        sm = xdataset['SM'].data
        clear = np.where((sm&1==0) & (sm&2==0) & (sm&4==0))
        clm[clear] = 0
        xdataset['clm'] = xarray.DataArray(clm, coords=[xdataset.Latitude, xdataset.Longitude], dims=['Latitude','Longitude'])


        if XTIE == SIZE1:
            to_float(xdataset,'SAA')
            to_float(xdataset,'SZA')
            to_float(xdataset,'VAA')
            to_float(xdataset,'VZA')
        else:
            VGT_TIE_COMPLETE(xdataset, 'SZA', (SIZE1,SIZE2), (XTIE,YTIE))
            VGT_TIE_COMPLETE(xdataset, 'SAA', (SIZE1,SIZE2), (XTIE,YTIE))
            VGT_TIE_COMPLETE(xdataset, 'VZA', (SIZE1,SIZE2), (XTIE,YTIE))
            VGT_TIE_COMPLETE(xdataset, 'VAA', (SIZE1,SIZE2), (XTIE,YTIE))

        lon,lat=np.meshgrid(xdataset['Longitude'],xdataset['Latitude'])
        xdataset['lat']=(('x', 'y'), lat)
        xdataset['lon']=(('x', 'y'), lon)

        # mean decimal time for the scene
        xdataset['mean-time'] = get_meantime(xdataset)
        xdataset['mean-time-dec'] = date_to_float(xdataset['mean-time'].data)

        for band in tab_band_internal:
            to_float(xdataset, band)
            bandunc = '{} uncertainty'.format(band)
            if  bandunc in xdataset.data_vars.keys():
                to_float(xdataset, bandunc)
            else:
                void = xarray.DataArray(np.zeros((SIZE1, SIZE2)) + np.NaN, coords=[xdataset.Latitude, xdataset.Longitude], dims=['Latitude','Longitude'])
                xdataset[bandunc] = void
            xdataset.rename({bandunc:'{}_unc'.format(band)}, inplace=True)

    elif sensor=='S3A':
        conv = {'lat':'y', 'lon':'x'}
        xdataset.rename(conv, inplace=True)
        conv = {'latitude': 'lat', 'longitude':'lon', 'cloud_an':'clm', 'sat_zenith_tn':'VZA','sat_azimuth_tn':'VAA'}
        xdataset.rename(conv, inplace=True)
        tab_band_internal = ['S1_radiance_an', 'S2_radiance_an', 'S3_radiance_an', 'S5_radiance_an']
        smac_coeff_name = ['smac_coeff_s3a_slstr_cont.npy']
        xdataset['mean-time'] = get_meantime_s3a(xdataset)
        xdataset['mean-time-dec'] = date_to_float(xdataset['mean-time'].data)
        for band in tab_band_internal:
            bandunc = band.replace('radiance','exception') # TODO: exception n'est pas un dataset erreur
            xdataset.rename({bandunc:'{}_unc'.format(band)}, inplace=True)
        SIZE1, SIZE2 = xdataset[tab_band_internal[0]].shape
        XTIE, YTIE = xdataset['SZA'].shape
        filtre = (np.isnan(xdataset['clm'].values))
        xdataset['clm'].values[filtre] = 0

    else:
        raise('unknow sensor')



#    try:
#        xdataset['mean-time']    = get_meantime(xdataset)
#    except:
#        time = basename(dirname(fname)).split('-')[3]
#        year = time[:4]
#        month = time[4:6]
#        day = time[6:8]
#        hour = time[8:10]
#        minute = time[10:12]
#        sec = time[12:]
#        if len(sec) == 1:
#            sec = '0{}'.format(sec)
#        xdataset['mean-time']  = np.datetime64(year + '-' + month + '-' + day + 'T' + hour + ':' + minute + ':' + sec)
#    xdataset['mean-time-dec']= date_to_float(xdataset['mean-time'].data)
                                                       
    return xdataset, SIZE1, SIZE2, tab_band_internal, smac_coeff_name

def save_h5(filename, data, rsurf, Drsurf, Drtoa, Duo3, Duh2o, Dpre, Dtaup, Irtoa, Iuo3, Iuh2o, Ipre, Itaup, Ialt, BREAKPOINT, INPUT, version):

    size = rsurf[0].shape

    out = h5py.File(filename, 'a')

    for att, value in data.attrs.items():
        out.attrs[att] = value
    out.attrs['date_created'] = str(datetime.now())
    out.attrs['production_center'] = 'hygeos'
    out.attrs['version'] = version

    dataset_names = ['TOC Bleu', 'TOC Red', 'TOC NIR', 'TOC SWIR']
    for idx in range(4):
        band = dataset_names[idx]
        out.create_dataset(band, size, dtype='float32', compression='gzip', compression_opts=9)
        out[band].attrs['Long_name'] = 'Top of Canopy Reflectance'
        out[band].attrs['Unit'] = 'None'
        out[band][:] = rsurf[idx]
        band = '{} error'.format(band)
        out.create_dataset(band, size, dtype='float32', compression='gzip', compression_opts=9)
        out[band].attrs['Long_name'] = 'Uncertainty Top of Canopy Reflectance'
        out[band].attrs['Unit'] = 'None'
        out[band][:] = Drsurf[idx]

    lon, lat = np.meshgrid(data['Longitude'].data, data['Latitude'].data)
    out.create_dataset('Lat', size, dtype='float32', compression='gzip', compression_opts=9)
    out['Latitude'].attrs['Unit'] = 'degree'
    out['Latitude'][:] = lat
    out.create_dataset('Lon', size, dtype='float32', compression='gzip', compression_opts=9)
    out['Longitude'].attrs['Unit'] = 'degree'
    out['Lontitude'][:] = lon

    out.create_dataset('SZA', size, dtype='float32', compression='gzip', compression_opts=9)
    out['SZA'].attrs['Unit'] = 'degree'
    out['SZA'][:] = data['SZA'].data
    out.create_dataset('SAA', size, dtype='float32', compression='gzip', compression_opts=9)
    out['SAA'].attrs['Unit'] = 'degree'
    out['SAA'][:] = data['SAA'].data
    out.create_dataset('VZA', size, dtype='float32', compression='gzip', compression_opts=9)
    out['VZA'].attrs['Unit'] = 'degree'
    out['VZA'][:] = data['VZA'].data
    out.create_dataset('VAA', size, dtype='float32', compression='gzip', compression_opts=9)
    out['VAA'].attrs['Unit'] = 'degree'
    out['VAA'][:] = data['VAA'].data

    out.create_dataset('SM', size, dtype='float32', compression='gzip', compression_opts=9)
#    out['SM'][:] = data['SM'].data
    out['SM'][:] = data['clm'].data

    if BREAKPOINT:
        out.create_dataset('Drtoa', Drtoa.shape, dtype='float32', compression='gzip', compression_opts=9)
        out.create_dataset('Duo3', Duo3.shape, dtype='float32', compression='gzip', compression_opts=9)
        out.create_dataset('Duh2o', Duh2o.shape, dtype='float32', compression='gzip', compression_opts=9)
        out.create_dataset('Dpre', Dpre.shape, dtype='float32', compression='gzip', compression_opts=9)
        out.create_dataset('Dtaup', Dtaup.shape, dtype='float32', compression='gzip', compression_opts=9)

    if INPUT:
        out.create_dataset('rtoa', Irtoa.shape, dtype='float32', compression='gzip', compression_opts=9)
        out.create_dataset('uo3',  Iuo3.shape, dtype='float32', compression='gzip', compression_opts=9)
        out.create_dataset('uh2o', Iuh2o.shape, dtype='float32', compression='gzip', compression_opts=9)
        out.create_dataset('pre', Ipre.shape, dtype='float32', compression='gzip', compression_opts=9)
        out.create_dataset('taup', Itaup.shape, dtype='float32', compression='gzip', compression_opts=9)
        out.create_dataset('alt', Ialt.shape, dtype='float32', compression='gzip', compression_opts=9)


    if BREAKPOINT:
        out['Drtoa'][:] = Drtoa
        out['Duo3'][:]  = Duo3
        out['Duh2o'][:] = Duh2o
        out['Dpre'][:]  = Dpre
        out['Dtaup'][:] = Dtaup
        
    if INPUT:
        out['rtoa'][:]  = Irtoa
        out['uo3'][:]   = Iuo3
        out['uh2o'][:]  = Iuh2o
        out['pre'][:]   = Ipre
        out['taup'][:]  = Itaup
        out['alt'][:]   = Ialt

    out.close()

def save_nc(filename, data, rsurf, Drsurf, version):
    print("save netCDF : {}".format(filename))
    size = rsurf[0].shape

    out = Dataset(filename, 'w', format='NETCDF4')

    for att, value in data.attrs.items():
        out.setncattr(att, value)

    out.date_created = str(datetime.now())
    out.production_center = 'hygeos'
    out.version = version

    width = size[1]
    height = size[0]
    h = out.createDimension('height', height)
    w = out.createDimension('width', width)

    dataset_names = ['TOC Blue', 'TOC Red', 'TOC NIR', 'TOC SWIR']
#    for idx in range(4):
    for idx in range(rsurf.shape[0]):
        band = dataset_names[idx]
        sds = out.createVariable(band, 'f', ('height','width'), complevel=9)
        sds[:] = rsurf[idx]
        sds.Long_name = 'Top of Canopy Reflectance'
        sds.Unit = 'None'
        band = '{} error'.format(band)
        sds = out.createVariable(band, 'f', ('height','width'), complevel=9)
        sds[:] = Drsurf[idx]
        sds.Long_name = 'Uncertainty Top of Canopy Reflectance'
        sds.Unit = 'None'

#    lon, lat = np.meshgrid(data['Longitude'].data, data['Latitude'].data)
    sds = out.createVariable('Lat', 'f', ('height','width'), complevel=9)
#    sds[:] = lat
    sds[:] = data['lat'].data
    sds.Unit = 'Degree'
    sds = out.createVariable('Lon', 'f', ('height', 'width'), complevel=9)
#    sds[:] = lon
    sds[:] = data['lon'].data
    sds.Unit = 'Degree'

    sds = out.createVariable('SZA', 'f', ('height', 'width'), complevel=9)
    sds[:] = data['SZA'].data
    sds.Unit = 'Degree'
    sds = out.createVariable('SAA', 'f', ('height', 'width'), complevel=9)
    sds[:] = data['SAA'].data
    sds.Unit = 'Degree'
    sds = out.createVariable('VZA', 'f', ('height', 'width'), complevel=9)
    sds[:] = data['VZA'].data
    sds.Unit = 'Degree'
    sds = out.createVariable('VAA', 'f', ('height', 'width'), complevel=9)
    sds[:] = data['VAA'].data
    sds.Unit = 'Degree'

    sds = out.createVariable('SM', 'f', ('height', 'width'), complevel=9)
#    sds[:] = data['SM'].data
    sds[:] = data['clm'].data

    out.close()

def main(filein, fileout, dem_lut, S):

#    path_i = '/rfs/data/C3S'
#    fname_i = '2001_2'
#
#    fname = '/rfs/data/C3S/2003_2/C3S-L1B-AVHRR_NOAA-2003040211561-fv0001.nc/testdata_Hornsund.nc'
#    fname = '/rfs/data/C3S/2003_2/C3S-L1B-AVHRR_NOAA-2003040211561-fv0001.nc/testdata_Ny_Alesund.nc'
    fname = filein

    # SMACG configuration
    XBLOCK = 512
    XGRID = 512
    YGRID = 1
    YBLOCK = 1
    NBLOOP = 1

    aer_coef = 'CONT'
    dir_coef_name = './'

    version = '1.0'

    k_uh2o = 1e-1
    k_uo3 = 1e-3
    k_p0 = 1e-2

    Etoa = 0
    ERtoa = 0.01
    Etaup = 0.05
    ERtaup = 0.15
    Euo3 = 0.0
    ERuo3 = 0.06
    Euh2o = 0.0
    ERuh2o = 0.2
    Epre = 1.0
    ERpre = 0.0


    data, SIZE1, SIZE2, tab_band_internal, smac_coeff_name = pre_image(fname, aer_coef)
    if data is None:
        print("l'image n'a pas d'attributs")
        return

#    bands_path = ["".join((dir_coef_name, 'COEFFS/'+x)) for x in smac_coeff_name]
#    print(bands_path)
#    coeffs     = get_smac_coeffs(bands_path)
#    print(coeffs, coeffs.shape)
#    exit(0)
    year = str(data['mean-time'].values)[:4]
    month = str(data['mean-time'].values)[5:7]
    day = str(data['mean-time'].values)[8:10]

    # path to input ancillary MERRA 2 data
    # 1) aerosols
    merra_aerosol='/rfs/data/MERRA2/aer_extinction/{0}/*MERRA2_*.tavg1_2d_aer_Nx.{0}{1}{2}*.nc*'.format(year, month, day)
    merra_aerosol=glob(merra_aerosol)[0]
    # 2) PTWO, Pressure, Temperature, Water vapour ,Ozone
    merra_ptwo='/rfs/data/MERRA2/surf_pression_water_vapor/{0}/*MERRA2_*.tavg1_2d_slv_Nx.{0}{1}{2}*.nc*'.format(year, month, day)
    merra_ptwo=glob(merra_ptwo)[0]

    merra_lut = pre_merra2(merra_aerosol, merra_ptwo)

    # files containg SMAC coefficients

    bands_path = ["".join((dir_coef_name, 'COEFFS/'+x)) for x in smac_coeff_name]
    # masking cloudy & out of orbit pixels and SZA above 90°
    SM   = data['clm'].data
    SZA  = data['SZA'].data
    good = np.where((SM == 0) & (SZA < 90))
    if len(good[0]) == 0:
        print('image nuageuse')
        return
#    good = np.where((SZA < 90))
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
            rtoa_err[iband,:] = data[band+'_unc'].data[good]

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
    taup550_ext[:GSIZE]  = taup550
    uo3_ext[:GSIZE]      = uo3
    pressure_ext[:GSIZE] = pressure
    uh2o_ext[:GSIZE]     = uh2o
    tetas_ext[:GSIZE]    = tetas
    tetav_ext[:GSIZE]    = tetav
    phis_ext[:GSIZE]     = phis
    phiv_ext[:GSIZE]     = phiv

    #Getting SMAC coefficients
    print(bands_path)
    coeffs     = get_smac_coeffs(bands_path)
    print(coeffs, coeffs.shape)

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

        inter[good]  = abs(Jrtoa_ext[i,:GSIZE] * rtoa_err[i,:]               )
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

    if fileout.split('.')[-1] == 'h5':
        save_h5(fileout, data, rsurf, Drsurf, Drtoa, Duo3, Duh2o, Dpre, Dtaup, Irtoa, Iuo3, Iuh2o, Ipre, Itaup, Ialt, BREAKPOINT, INPUT, version)
    elif fileout.split('.')[-1] == 'nc':
        save_nc(fileout, data, rsurf, Drsurf, version)


if __name__=='__main__':
#    filein = '/rfs/data/C3S/2003_2/C3S-L1B-AVHRR_NOAA-20030719130848-fv0001.nc/testdata_Efri-Vik_Iceland.nc'
#    fileout = '/rfs/proj/C3S/testdata/2003_2/C3S-L1B-AVHRR_NOAA-20030719130848-fv0001.nc/testdata_Efri-Vik_Iceland.h5'
    mep = True
    typeout = 'netcdf4' #'hdf5'
    # avhrr
    path_i = '/rfs/data/C3S'
    path_o = '/rfs/proj/C3S/testdata'
    # vgt
#    path_i = '/rfs/data/C3S/VGT/EXTRACT_V2/'
#    path_o = '/rfs/proj/C3S/testdata_VGT/'
    path_i = '/rfs/data/VGT/VGTP_extracts_49x49_180129'
    path_o = '/rfs/proj/C3S/VGTP_extracts_49x49_lev2'

    path_i = '/rfs/data/AVHRR/group2'
    path_o = '/rfs/proj/C3S/AVHRR_group2_lev2'

    path_i = '/rfs/data/C3S/C3S_312a_Lot9/L2B_VGT_AERONET'
    path_o = '/rfs/user/bruno/C3S/test'
    year = 2002
    # test VGT
#    filein = '/rfs/data/C3S/VGT/EXTRACT_V2/189_Gozo_V220050601037.h5'
#    filein = '/rfs/data/C3S/VGT/EXTRACT_V2/33_IMC_Oristano_V220050601036.h5'
#    fileout = './test.h5'

    if not(mep):
        fdem = '/rfs/data/DEM/GLOBE/GTOPO30_DZ_MLUT.nc'
        dem_lut = read_mlut(fdem)

    S = Smaccl('CPU')

    files = glob('{}/{}/*/*.nc'.format(path_i,year))
    files = ['/rfs/user/bruno/C3S/input/C3S_L2A_20030801_X22Y04_339_1KM_VGT_V001.h5']
    files = ['/rfs/data/VGT/VGTP_extracts_49x49_180129/1999/19990601/11_Ispra_V119990601138.h5']

    files = ['/rfs/proj/C3S/Level2_vito/S3A_TEST____20180703.SEN3_syn.nc']
    files = ['/rfs/user/bruno/C3S/input/cgl_SEN3-SYN-BS-RP_201807030950_AE-Carpentras_S3A_v0.0.rc2.nc']
    files = ['/rfs/user/bruno/C3S/input/cgl_SEN3-SYN-BS-RP_201807030950_X18Y03_S3A_v0.0.rc2.nc']
#    for filein in glob('{}/{}/*/*.h5'.format(path_i,year)):
    for filein in files:
        print(filein)
        # read dem
        if mep:
            tile = basename(filein).split('_')[3]
            fdem = glob('/rfs/proj/C3S/ancillary/dem/*{}.nc'.format(tile))[0]
            dem_lut = read_mlut(fdem)
        dirout = '{}/{}/{}'.format(path_o, year, basename(dirname(filein)))
        if not(exists(dirout)):
            system('mkdir -p {}'.format(dirout))
        if typeout == 'hdf5':
            fileout = '{}/{}'.format(dirout, basename(filein))
        else:
            fileout = '{}/{}.nc'.format(dirout, basename(filein)[:-3])
        fileout = '/rfs/user/bruno/C3S/test.nc'
#        if exists(fileout):
#            print("skip")
#            continue
        main(filein, fileout, dem_lut, S)
        break
#    for dirdate in glob('{}/*'.format(path_i)):
#        if basename(dirdate)[:4] == '2003':
#            for subdir in glob('{}/*'.format(dirdate)):
#                if exists(basename(subdir)):
#                    continue
#                for filein in glob('{}/*'.format(subdir)):
#                    print(filein)
#                    if filein[-3:] == '.nc':
#                        dirout = '{}/{}/{}'.format(path_o, basename(dirdate), basename(subdir))
#                        if not(exists(dirout)):
#                            system('mkdir -p {}'.format(dirout))
#                        fileout = '{}/{}.h5'.format(dirout, basename(filein)[:-3])
#                        if exists(fileout):
#                            continue
#                        print('{} => {}'.format(filein, fileout))
#                        main(filein, fileout, dem_lut, S)

#    main(filein, fileout, dem_lut, S)
