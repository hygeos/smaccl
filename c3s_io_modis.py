import xarray as xa
import numpy as np
from smaccl import get_smac_coeffs, type_coeff_reduced, type_coeff
from datetime import datetime
from c3s_lib import SRF, date_to_float

def set_smac_indice(file, bands):
    data = np.load(file)
    bandnames = []
    for d in data['bandname'][:,0]:
        if len(d) == 0: 
            bandnames.append('')
        else:
#            bandnames.append(d.split('_')[8])
#            bandnames.append(d.split()[-1])
            bandnames.append(d.split(',')[-1].split()[0])
    indices = []
    for b in bands:
#        tmp = '{}{}'.format(b[0], int(b[1:]))
        idx = np.where(b==np.array(bandnames))[0][0]
        indices.append(idx)

    return indices

def load_modis(filename: str, smacdir, chunkidx, chunksize, bands):

    if (chunksize < 0):
        yslice = slice(None, None)
    else:
        ymin = chunkidx * chunksize
        ymax = ymin + chunksize
        yslice= slice(ymin,ymax)

    ds = xa.open_mfdataset(filename).squeeze('time').drop_vars('time')
    
    glon2D, glat2D = np.meshgrid(ds['lon'].values, ds['lat'].values)
    glat2D = glat2D.astype('float32')
    glon2D = glon2D.astype('float32')

    lat2D = glat2D[yslice, :]
    lon2D = glon2D[yslice, :]
    
    gl_size = lat2D.shape # global shape 
    
    if chunksize == -1:
        xdataset = xa.Dataset()
        global_attrs = ds.attrs
        global_attrs['title'] = '{} MODIS TOC reflectance'.format(ds.platform)
        xdataset = xdataset.assign_attrs(global_attrs)
        return xdataset, None, gl_size, None
    
    # process reflectance for each band (modify inplace)
    #for band in bands: 
    bands_smac = [] 
    bands_vito = []
#    for iband, band in enumerate(bands):
#        bands_vito.append('EV_RefSB_{}'.format(band))
#        ds[bands_vito[iband]] = ds[bands_vito[iband]] * ds[bands_vito[iband]].reflectance_scale / (np.cos(np.deg2rad(ds.solar_zenith)) * ds[bands_vito[iband]].radiance_scale)
#        bands_smac.append('eos_02_modis_{:02d}.flt'.format(int(band)))
#        band_unc = '{}_uncert'.format(bands_vito[iband])
#        ds['{}_err'.format(band)] = ds[band_unc]*ds[bands_vito[iband]]/100


#    sza = ds['solar_zenith'].values[yslice, :]
#    vza = ds['sensor_zenith'].values[yslice, :]
#    saa = ds['solar_azimuth'].values[yslice, :]
#    vaa = ds['sensor_azimuth'].values[yslice, :]
    sza =   ds['Solar_Zenith_Angle'].values[yslice, :]
    vza =  ds['Sensor_Zenith_Angle'].values[yslice, :]
    saa =  ds['Solar_Azimuth_Angle'].values[yslice, :]
    vaa = ds['Sensor_Azimuth_Angle'].values[yslice, :]

    names_conv = [('Solar_Zenith_Angle','SZA'), ('Sensor_Zenith_Angle','VZA'), ('Solar_Azimuth_Angle','SAA'), ('Sensor_Azimuth_Angle','VAA')]

    #--------------------------------------------------------
    # /!\ WARNING /!\ TODO review this strategy:
    #--------------------------------------------------------
#    cloud = ds['Integer_Cloud_Mask']
#    cloud = ds['Cloud_Mask']
#    cloud = cloud != 3          # clouds = everything not 'confidently clear'
#    cloud = cloud[yslice,:]  # get region
    # 0 = cloudy, 
    # 1 = probably cloudy, 
    # 2 = probably clear, 
    # 3 = confident clear, -1 = no result)
    #--------------------------------------------------------

    xdataset = xa.Dataset({
        'SZA': (['y','x'], sza), 
        'VZA': (['y','x'], vza), 
        'SAA': (['y','x'], saa),
        'VAA': (['y','x'], vaa), 
        'lat': (['y','x'], lat2D.data), 
        'lon': (['y','x'], lon2D.data), 
#        'clm': (['y','x'], cloud.data)}
    }
        )

    variables_exclusion = ['crs','lon', 'lat', 'Sensor_Azimuth_Angle', 'Sensor_Zenith_Angle', 'Solar_Azimuth_Angle', 'Solar_Zenith_Angle', 
                           'EV_RefSB_1', 'EV_RefSB_2', 'EV_RefSB_3', 'EV_RefSB_4', 'EV_RefSB_5', 'EV_RefSB_7', 'EV_RefSB_1_Uncert', 'EV_RefSB_2_Uncert', 
                           'EV_RefSB_3_Uncert', 'EV_RefSB_4_Uncert', 'EV_RefSB_5_Uncert', 'EV_RefSB_7_Uncert', '1_err', '2_err', '3_err','4_err','5_err','7_err'] #, 'Cloud_Mask']

    bands_type = {'nnrow_500m':'int32','nncol_500m':'int32','nndist_500m':'float32','nnrow_1km':'int32','nncol_1km':'int32','nndist_1km':'float32','Cloud_Mask':'uint8'}

    for v in ds.variables:
        if v in variables_exclusion:
            continue
#        data = ds[v].values[yslice,:]
        data = ds[v].data[yslice,:]
        data[np.isnan(data)] = -1
        data = data.astype(bands_type[v])
        xdataset[v] = xa.DataArray(data=data, attrs=ds[v].attrs, dims=['y','x'])
    global_attrs = ds.attrs
    global_attrs['title'] = '{} MODIS TOC reflectance'.format(ds.platform)
    xdataset = xdataset.assign_attrs(global_attrs)
#    xdataset = xdataset.assign_coords(ds.coords)
    for n_l1, n_l2 in names_conv:
        xdataset[n_l2] = xdataset[n_l2].assign_attrs(ds[n_l1].attrs)

    for iband, band in enumerate(bands):
        bands_vito.append('EV_RefSB_{}'.format(band))
        bands_smac.append('eos_01_modis_{:02d}.flt'.format(int(band)))
        reflectances =  ds[bands_vito[iband]].values[yslice,:] * ds[bands_vito[iband]].reflectance_scale / (np.cos(np.deg2rad(ds.Solar_Zenith_Angle.values[yslice,:]))  * ds[bands_vito[iband]].radiance_scale)
        xdataset[bands_vito[iband]] = (['y','x'], reflectances)
        # /!\ Unsure about the division (would suggest percentage data, but current data E [0, 1])
#        xdataset['{}_err'.format(bands_vito[iband])] = (['y','x'], ds['{}_uncert'.format(bands_vito[iband])].values[yslice,:]*ds[bands_vito[iband]].values[yslice,:]/100)
        xdataset['{}_err'.format(bands_vito[iband])] = (['y','x'], ds['{}_Uncert'.format(bands_vito[iband])].values[yslice,:]*reflectances/100)

    platform_smac = {'Aqua':'EOS_2','Terra':'EOS_1'}
    smacfile = '{}/{}_MODIS_smac_coeffsi_v3.0.npy'.format(smacdir, platform_smac[ds.platform])
    idx_smac = set_smac_indice(smacfile, bands_smac)
    coeff_smac = get_smac_coeffs(smacfile, idx_smac)


    # compute mean datetime 
    # str -> datetime obj
    dt_s = datetime.strptime(ds.attrs['time_coverage_start'], '%Y-%m-%dT%H:%M:%SZ')
    dt_e = datetime.strptime(ds.attrs['time_coverage_end'], '%Y-%m-%dT%H:%M:%SZ')
    # compute mean of start and end
    mean_time = np.datetime64(dt_s + (dt_e - dt_s) / 2)
    xdataset['mean-time'] = mean_time
    xdataset['mean-time-dec'] = date_to_float(mean_time)

    return xdataset, coeff_smac, gl_size, bands_vito