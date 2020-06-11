from netCDF4 import Dataset
import xarray as xa
import numpy as np
from smaccl import get_smac_coeffs, type_coeff_reduced, type_coeff
from c3s_lib import SRF, date_to_float
from datetime import datetime
import sys
sys.path.insert(0,'./eoread')
from eoread.msi import Level1_MSI
from eoread.landsat8_oli import Level1_L8_OLI
from os.path import basename

def load_testcase_vito(fname, dirsmac, smac_version, sensor):
    data = Dataset(fname)

    lat_axis = data['lat'][:]
    lon_axis = data['lon'][:]
    SIZE1 = len(lon_axis)
    SIZE2 = len(lat_axis)
    gl_size = (SIZE1, SIZE2)

    sza = np.reshape(data['sza'][:], gl_size)
    vza_vnir = np.reshape(data['vza_vnir'][:], gl_size)
    vza_swir = np.reshape(data['vza_swir'][:], gl_size)
    saa = np.reshape(data['saa'][:], gl_size)
    vaa_vnir = np.reshape(data['vaa_vnir'][:], gl_size)
    vaa_swir = np.reshape(data['vaa_swir'][:], gl_size)

    lon, lat = np.meshgrid(lon_axis, lat_axis)

    if sensor == 'AVHRR':
        cloud = np.logical_not((np.reshape(data['sm'][:], gl_size).astype('int')&15 == 8)).astype('int')
        sm = np.reshape(data['sm'], gl_size)
        hour = basename(fname).split('_')[-5]
        platform = basename(fname).split('_')[-2]
        smacfile = '{}/{}_{:02d}_smac_coeffs_v{}.npy'.format(dirsmac, platform[:4], int(platform[4:]), smac_version)
        if int(platform[4:]) < 15:
            bandnames = ['band1','band2']
        else:
            bandnames = ['band1','band2','band3']
    elif sensor == 'PROBAV':
        bandnames = ['band1','band2','band3','band4']
        if 'sm' in data.variables:
            varname = 'sm'
            hour = basename(fname).split('_')[-5]
        else:
            varname = 'SM'
            hour = basename(fname).split('_')[-4]
        cloud = np.logical_not((np.reshape(data[varname][:], gl_size).astype('int')&15 == 8)).astype('int')
        sm = np.reshape(data[varname], gl_size)
        smacfile = '{}/PROBA-V_smac_coeffs_v{}.npy'.format(dirsmac, smac_version)
        if 'camera' in data.ncattrs():
            camera = data.getncattr('camera')
            cam = {'1':'LEFT', '2':'CENTER', '3':'RIGHT'}
            smacfile = '{}/PROBA-V_{}_smac_coeffs_v{}.npy'.format(dirsmac, cam[camera], smac_version)

    elif sensor == 'VGT':
        bandnames = ['band1','band2','band3','band4']
        cloud = np.logical_not((np.reshape(data['sm'][:], gl_size).astype('int')&15 == 8)).astype('int')
        sm = np.reshape(data['sm'], gl_size)
        hour = str(datetime.strptime(data.time_coverage_start, '%Y/%m/%d %H:%M:%S') + (datetime.strptime(data.time_coverage_end, '%Y/%m/%d %H:%M:%S') - datetime.strptime(data.time_coverage_start, '%Y/%m/%d %H:%M:%S'))/2).split()[-1].replace(':','')
        smacfile = '{}/VGT{}_smac_coeffs_v{}.npy'.format(dirsmac, data.sensor.split('-')[-1], smac_version)

    date = basename(fname).split('_')[2]
    date = "{}-{}-{}".format(date[:4], date[4:6], date[6:8])
    hour = "{}:{}:{}".format(hour[:2], hour[2:4], hour[4:6])
    dt = np.datetime64('{}T{}'.format(date,hour))

    xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza_vnir), 'VAA': (['y','x'], vaa_vnir), 'VAA_SWIR': (['y','x'], vaa_swir), 'VZA_SWIR': (['y','x'], vza_swir), 
                           'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 
                           'clm':(['y','x'], cloud.astype('float32')), 'sm':(['y','x'], sm)}, 
                           coords={'x':(['x'], np.array(lon_axis)), 'y':(['y'], np.array(lat_axis))})


    for b in bandnames:
        xdataset[b] = (['y','x'], np.reshape(data[b][:], gl_size))
        berr = '{}_err'.format(b)
        xdataset[berr] = (['y','x'], np.reshape(data[berr][:]*data[b][:]*.01, gl_size))

    xdataset['mean-time'] = dt
    xdataset['mean-time-dec'] = date_to_float(xdataset['mean-time'].data)

    smacdata = np.load(smacfile)
    coeffs = np.zeros((len(bandnames), smacdata.shape[1]), dtype=type_coeff_reduced, order='C')
    for idx in range(len(bandnames)):
        for i2, d in enumerate(smacdata[idx]):
            coeffs[idx,i2] = d.tolist()[1:]

    return xdataset, SIZE1, SIZE2, bandnames, coeffs, gl_size


def load_olci_slstr(fname, smacfile, chunkidx, chunksize, platform='S3A', bands_olci=None, bands_slstr=None):

    _,_,_,wvl_central_olci,_,_,_  = SRF(platform+'_OLCI')
    _,_,_,wvl_central_slstr,_,_,_ = SRF(platform+'_SLSTR')
    wav = {'olci': wvl_central_olci, 'slstr': wvl_central_slstr}  

    if (chunksize < 0):
        ymin=0
        ymax=-1
    else:
        ymin = chunkidx*chunksize
        ymax = ymin + chunksize
    pfile = Dataset(fname)
    date = pfile.getncattr('start_date')

    lat = pfile['latitude'][ymin:ymax,:].astype('float32')
    lon = pfile['longitude'][ymin:ymax,:].astype('float32')
    cloud = pfile['cloud_an'][ymin:ymax,:]
    quality_fg = pfile['quality_flags'][ymin:ymax,:]
    pxl_classif_fg = pfile['pixel_classif_flags'][ymin:ymax,:]

    vza = pfile['OZA'][ymin:ymax,:]
    vaa = pfile['OAA'][ymin:ymax,:]
    sza = pfile['SZA'][ymin:ymax,:]
    saa = pfile['SAA'][ymin:ymax,:]

    mus = np.cos(sza*np.pi/180.)
    if bands_olci is None:
        olci_idx = [2,3,4,5,6,7,8,9,10,11,12,16,17,18,21]
    else:
        olci_idx = bands_olci

    if 'lat_intern' in pfile.variables.keys():
        lat_axis = pfile['lat_intern']
        lon_axis = pfile['lon_intern']
    else:
        lat_axis = pfile['lat']
        lon_axis = pfile['lon']

    xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza), 'VAA': (['y','x'], vaa), 
                           'cloud_an': (['y','x'], cloud), 'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 
                           'quality_flags':(['y','x'], quality_fg.astype('int32')), 'pixel_classif_flags':(['y','x'], pxl_classif_fg), 
                           'clm':(['y','x'], cloud.astype('float32'))}, 
                           coords={'x':(['x'], lon_axis[:]), 'y':(['y'], lat_axis[ymin:ymax])})

    tab_band_internal = []
    central_wvl = []
    sensor= []
    # bands olci
    for idx in olci_idx:
        rad_band = 'Oa{:02d}_radiance'.format(idx)
        tab_band_internal.append(rad_band)
        central_wvl.append(wav['olci'][idx-1])
        sensor.append('olci')
        ltoa = pfile[rad_band][ymin:ymax,:]
        f0_band = 'solar_flux_band_{}'.format(idx)
        f0 = pfile[f0_band][ymin:ymax, :]

        rtoa = (np.pi*ltoa)/(mus*f0)
        xdataset[rad_band] = (['y','x'], rtoa)

    coeff_olci = get_smac_coeffs(smacfile['olci'], np.array(olci_idx)-1)

    # band slstr
    if bands_slstr is None:
        slstr_idx = [1,2,3,4,5,6]
    else:
        slstr_idx = bands_slstr

    for idx in slstr_idx:
        s_band = 'S{}_radiance_an'.format(idx)
        if idx!=4:
            tab_band_internal.append(s_band)
            central_wvl.append(wav['slstr'][idx-1])
            sensor.append('slstr')
        ltoa = pfile[s_band][ymin:ymax,:]
        f0 = pfile[s_band].getncattr('solar_irradiance')[0]
        rtoa = (np.pi*ltoa)/(mus*f0)
        xdataset[s_band] = (['y','x'], rtoa)

    SIZE1, SIZE2 = xdataset[tab_band_internal[0]].shape
    if 4 in slstr_idx : slstr_idx.remove(4)
    coeff_slstr = get_smac_coeffs(smacfile['slstr'], np.array(slstr_idx)-1)

    coeff_smac = np.concatenate([coeff_olci, coeff_slstr])

    day =   date[:2]
    year =  date[7:11]
    month = date[3:6]
    hour =  date[12:14]
    minu =  date[15:17]
    sec =   date[18:20]
    strmonths = np.array(['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])
    month = np.where(month==strmonths)[0][0]
    dt = np.datetime64(year + '-' + '{:02d}'.format(month+1) + '-' + day + 'T' + hour + ':' + minu + ':' + sec)
    xdataset['mean-time'] = dt
    xdataset['mean-time-dec'] = date_to_float(xdataset['mean-time'].data)

    filtre = (np.isnan(xdataset['clm'].values))
    xdataset['clm'].values[filtre] = 0

    gl_size = pfile['latitude'][:].shape

    pfile.close()

    return xdataset, SIZE1, SIZE2, tab_band_internal, central_wvl, sensor, coeff_smac, gl_size


def load_msi(fname, smacfile, chunkidx, chunksize, 
        platform='S2A', bands_msi=None, remove_blank=True, split=True, resolution='10'):

    _,_,_,wvl_central_msi,_,_,_  = SRF(platform+'_MSI')
    wav = {'msi': wvl_central_msi} 

    pfile = Level1_MSI(fname, split=split, resolution=resolution)
    date  = pfile.attrs['datetime']
    bnames=[]
    for ds in pfile:
        if 'Rtoa' in ds:
            bnames.append(ds)
    if remove_blank:
        good = np.where(pfile[bnames[0]] != 0.)
        gslicex = slice(good[1][0], good[1][-1])
        gslicey = slice(good[0][0], good[0][-1])
        pfile   = pfile.sel(columns=gslicex, rows=gslicey)

    if (chunksize < 0):
        yslice=slice(None)
    else:
        ymin  = chunkidx*chunksize
        ymax  = ymin + chunksize
        yslice= slice(ymin,ymax)

    lat = pfile['latitude'][yslice,:].astype('float32')
    lon = pfile['longitude'][yslice,:].astype('float32')
    cloud = np.zeros_like(lat)

    vza = pfile['vza'][yslice,:]
    vaa = pfile['vaa'][yslice,:]
    sza = pfile['sza'][yslice,:]
    saa = pfile['saa'][yslice,:]
    if bands_msi is None:
        msi_idx = list(np.arange(13)+1)
    else:
        msi_idx = bands_msi

    xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza), 'VAA': (['y','x'], vaa), 
                           'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 'clm':(['y','x'], cloud)})

    tab_band_internal = []
    central_wvl = []
    sensor = []

    for idx in msi_idx:
        rad_band = bnames[idx-1] 
        tab_band_internal.append(rad_band)
        central_wvl.append(wav['msi'][idx-1])
        sensor.append('msi')
        rtoa = pfile[rad_band][yslice,:]
        xdataset[rad_band] = (['y','x'], rtoa)

    coeff_smac = get_smac_coeffs(smacfile['msi'], np.array(msi_idx)-1)

    dt = np.datetime64(date)
    xdataset['mean-time'] = dt
    xdataset['mean-time-dec'] = date_to_float(dt)

    gl_size = pfile['latitude'].shape

    pfile.close()
    SIZE1, SIZE2 = xdataset[tab_band_internal[0]].shape

    return xdataset, SIZE1, SIZE2, tab_band_internal, central_wvl, sensor, coeff_smac, gl_size


def load_oli(fname, smacfile, chunkidx, chunksize, 
        platform='LANDSAT8', bands_oli=None, remove_blank=True, split=True):

    _,_,_,wvl_central_oli,_,_,_  = SRF(platform+'_OLI')
    wav = {'oli': wvl_central_oli} 

    pfile = Level1_L8_OLI(fname, split=split, l8_angles='./l8_angles/l8_angles')
    date  = pfile.attrs['datetime']
    gl_size = (int(pfile['totalheight'].data), int(pfile['totalwidth'].data))
    bnames=[]
    for ds in pfile:
        if 'Rtoa' in ds:
            bnames.append(ds)
    if remove_blank:
        good = np.where(pfile[bnames[0]] > 0)
        if len(good[0]) == 0:
            return None, None, None, None, None, None, None, gl_size 
        gslicex = slice(good[1][0], good[1][-1])
        gslicey = slice(good[0][0], good[0][-1])
        pfile   = pfile.sel(columns=gslicex, rows=gslicey)

    if (chunksize < 0):
        yslice=slice(None)
    else:
        ymin  = chunkidx*chunksize
        ymax  = ymin + chunksize
        yslice= slice(ymin,ymax)

    lat = pfile['latitude'][yslice,:].astype('float32')
    lon = pfile['longitude'][yslice,:].astype('float32')
    cloud = np.zeros_like(lat)

    vza = pfile['vza'][yslice,:]
    vaa = pfile['vaa'][yslice,:]
    sza = pfile['sza'][yslice,:]
    saa = pfile['saa'][yslice,:]
    if bands_oli is None:
        oli_idx = list(np.arange(7)+1)
    else:
        oli_idx = bands_oli

    xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza), 'VAA': (['y','x'], vaa), 
                           'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 'clm':(['y','x'], cloud)})

    tab_band_internal = []
    central_wvl = []
    sensor = []

    for idx in oli_idx:
        rad_band = bnames[idx-1] 
        tab_band_internal.append(rad_band)
        central_wvl.append(wav['oli'][idx-1])
        sensor.append('oli')
        rtoa = pfile[rad_band][yslice,:]
        xdataset[rad_band] = (['y','x'], rtoa)

    coeff_smac = get_smac_coeffs(smacfile['oli'], np.array(oli_idx)-1)

    dt = np.datetime64(date)
    xdataset['mean-time'] = dt
    xdataset['mean-time-dec'] = date_to_float(dt)

    gl_size = pfile['latitude'].shape

    pfile.close()
    SIZE1, SIZE2 = xdataset[tab_band_internal[0]].shape

    return xdataset, SIZE1, SIZE2, tab_band_internal, central_wvl, sensor, coeff_smac, gl_size


def create_nc(filename, gl_size, attrs, version):
    '''
    Start netCDF output file creation
    Inputs:
        filename : string of absolute path
        gl_size  : a tuple containing height and width
        attrs    : a list oa attributs
        version  : string containing version

    Outputs:
        a NETCDF4 Dataset
    '''
    print("save netCDF : {}".format(filename))
    out = Dataset(filename, 'w', format='NETCDF4')

    for att, value in attrs:
        out.setncattr(att, value)

    out.date_created = str(datetime.now())
    out.production_centre = 'vito'
    out.version = version

    width = gl_size[1]
    height = gl_size[0]

    out.createDimension('height', height)
    out.createDimension('width', width)

    return out


def save_nc(out, data, rsurf, Drsurf, version, dataset_names, chunkidx, chunksize, ancillary=None, sensor=None): 
    '''
    Save outputs
    Inputs:
        out : a NetCDF4 Dataset    
        data: the xarray containing Level 1 data 
        rsurf:  the numpy array containing the TOC reflectance
        Drsurf:  the numpy array containing the uncertainty on TOC reflectance
        version  : string containing version
        dataset_names : the list of string containing the names of the bands in level 1 file
        chunkidx: the number of the chunk to be saved
        chunksize: size of the chunk
   '''
    if (chunksize < 0):
        yslice=slice(None)
    else:
        ymin  = chunkidx*chunksize
        ymax  = ymin + chunksize
        yslice= slice(ymin,ymax)

    # test if some chunks have already been saved in the output file
    create  = not ('Lat' in out.variables)

    for idx in range(rsurf.shape[0]):
        band = dataset_names[idx].replace('_radiance','').replace('Rtoa_','')
        if create : sds = out.createVariable('TOC_{}'.format(band), 'f', ('height','width'), complevel=9)
        else: sds = out['TOC_{}'.format(band)]
        sds[yslice, :] = rsurf[idx]
        sds.Long_name = 'Top of Canopy Reflectance'
        sds.Unit = 'None'
        band = 'TOC_{} error'.format(band)
        if create: sds = out.createVariable(band, 'f', ('height','width'), complevel=9)
        else: sds = out[band]
        sds[yslice,:] = Drsurf[idx]
        sds.Long_name = 'Uncertainty Top of Canopy Reflectance'
        sds.Unit = 'None'
    if create: sds = out.createVariable('Lat', 'f', ('height','width'), complevel=9)
    else: sds = out['Lat']
    sds[yslice,:] = data['lat'].data
    sds.Unit = 'Degree'
    if create: sds = out.createVariable('Lon', 'f', ('height', 'width'), complevel=9)
    else: sds = out['Lon']
    sds[yslice,:] = data['lon'].data
    sds.Unit = 'Degree'
    
    if create: sds = out.createVariable('SZA', 'f', ('height', 'width'), complevel=9)
    else: sds = out['SZA']
    sds[yslice,:] = data['SZA'].data
    sds.Unit = 'Degree'
    if create : sds = out.createVariable('SAA', 'f', ('height', 'width'), complevel=9)
    else: sds = out['SAA']
    sds[yslice,:] = data['SAA'].data
    sds.Unit = 'Degree'
    if sensor=='PROBAV':
        if create: sds = out.createVariable('VZA_VNIR', 'f', ('height', 'width'), complevel=9)
        else: sds = out['VZA_VNIR']
        sds[yslice,:] = data['VZA'].data
        sds.Unit = 'Degree'
        if create: sds = out.createVariable('VAA_VNIR', 'f', ('height', 'width'), complevel=9)
        else: sds = out['VAA_VNIR']
        sds[yslice,:] = data['VAA'].data
        sds.Unit = 'Degree'

        if create: sds = out.createVariable('VZA_SWIR', 'f', ('height', 'width'), complevel=9)
        else: sds = out['VZA_SWIR']
        sds[yslice,:] = data['VZA_SWIR'].data
        sds.Unit = 'Degree'
        if create: sds = out.createVariable('VAA_SWIR', 'f', ('height', 'width'), complevel=9)
        else: sds = out['VAA_SWIR']
        sds[yslice,:] = data['VAA_SWIR'].data
        sds.Unit = 'Degree'
    else:
        if create: sds = out.createVariable('VZA', 'f', ('height', 'width'), complevel=9)
        else: sds = out['VZA']
        sds[yslice,:] = data['VZA'].data
        sds.Unit = 'Degree'
        if create: sds = out.createVariable('VAA', 'f', ('height', 'width'), complevel=9)
        else: sds = out['VAA']
        sds[yslice,:] = data['VAA'].data
        sds.Unit = 'Degree'

    if ancillary is not None:
        if create: sds = out.createVariable('uo3', 'f', ('height', 'width'), complevel=9)
        else: sds = out['uo3']
        sds[yslice,:] = ancillary[0]
        if create: sds = out.createVariable('uh2o', 'f', ('height', 'width'), complevel=9)
        else: sds = out['uh2o']
        sds[yslice,:] = ancillary[1]
        if create: sds = out.createVariable('aot550', 'f', ('height', 'width'), complevel=9)
        else: sds = out['aot550']
        sds[yslice,:] = ancillary[2]
        if create: sds = out.createVariable('iaer', 'u4', ('height', 'width'), complevel=9)
        else: sds = out['iaer']
        sds[yslice,:] = ancillary[3]
        if create: sds = out.createVariable('pressure', 'f', ('height', 'width'), complevel=9)
        else: sds = out['pressure']
        sds[yslice,:] = ancillary[4]
        if create: sds = out.createVariable('alt', 'f', ('height', 'width'), complevel=9)
        else: sds = out['alt']
        sds[yslice,:] = ancillary[5]

    # test mask
    if create: sds = out.createVariable('sm', 'u4', ('height', 'width'), complevel=9)
    else: sds = out['sm']
    sds[yslice,:] = data['sm']
    # end test

    create2  = 'cloud_an' in data.variables
    if create2 : 
        if create: sds = out.createVariable('cloud_an', 'f', ('height', 'width'), complevel=9)
        else: sds = out['cloud_an']
        sds[yslice,:] = data['cloud_an']
        if create: sds = out.createVariable('quality_flags', 'u4', ('height', 'width'), complevel=9)
        else: sds = out['quality_flags']
        sds[yslice,:] = data['quality_flags']#.data.astype('uint32')
        if create: sds = out.createVariable('pixel_classif_flags', 'u2', ('height', 'width'), complevel=9)
        else: sds = out['pixel_classif_flags']
        sds[yslice,:] = data['pixel_classif_flags']#.data.astype('uint16')
        if create: sds = out.createVariable('AC_process_flag', 'u1', ('height','width'), complevel=9)
        else: sds = out['AC_process_flag']
        sds[yslice,:] = data['ac_process_flag']#.data.astype('uint8')

    if create: sds = out.createVariable('ac_flag', 'u4', ('height','width'), complevel=9)
    else: sds = out['ac_flag']
    sds[yslice,:] = data['ac_flag']
