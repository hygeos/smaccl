from netCDF4 import Dataset
import xarray as xa
import numpy as np
#from smaccl import get_smac_coeffs
from matplotlib.pyplot import imshow, show
from utils import get_smac_coeffs

def date_to_float(d, epoch=np.datetime64('1980-01-01T00:00:00.000000000')):
    '''
        transform the date into a duration in minutes since epoch
            '''
    return (d - epoch).astype(np.float64)/1.0e9/60.

def load(fname, smacfile, bandidx, bandsize, bands_olci=None, bands_slstr=None):

    wav = {'olci': np.array([ 400.37414551,  411.93795776,  443.01525879,  490.36950684,
         510.35684204,  560.3480835 ,  620.2520752 ,  665.10235596,
         673.85638428,  681.37701416,  708.96862793,  754.02685547,
         761.55950928,  764.69177246,  767.82214355,  779.08508301,
         865.42785645,  884.18774414,  899.1463623 ,  939.18139648,
        1012.76092529]),
       'slstr': np.array([ 555.47,  660.1 ,  867.6 , 1375.1 , 1612.1 , 2253.  ])}  

    ymin = bandidx*bandsize
    ymax = ymin + bandsize
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

    xdataset = xa.Dataset({'SZA':(['y','x'], sza), 'SAA':(['y','x'], saa), 'VZA': (['y','x'], vza), 'VAA': (['y','x'], vaa), 'cloud_an': (['y','x'], cloud), 'lat': (['y','x'], lat), 'lon': (['y','x'], lon), 'quality_flags':(['y','x'], quality_fg.astype('int32')), 'pixel_classif_flags':(['y','x'], pxl_classif_fg), 'clm':(['y','x'], cloud.astype('float32'))}, 
            coords={'x':(['x'], lon_axis[:]),
                'y':(['y'], lat_axis[ymin:ymax])})
#            coords={'x':(['x'], pfile['lon'][:]),
#                'y':(['y'], pfile['lat'][ymin:ymax])})

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

    coeff_olci = get_smac_coeffs(smacfile['olci'], np.array(olci_idx))

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
    coeff_slstr = get_smac_coeffs(smacfile['slstr'], np.array(slstr_idx))

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

