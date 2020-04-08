import numpy as np
from glob import glob
import h5py
import sys
from luts.luts import MLUT
import xarray
sys.path.insert(0, '/home/did/RTC/SMART-G/')
from smartg.atmosphere import rod, simps

def SRF(sensor=None, camera=None):
    '''
    Arguments:
        sensor : one sensor name in the list return by SRF()
        camera : eventually a camera name (str) for a sensor
        
    returns:
        (wvn_limits, wvl_limits, fwhm, wvl_central, rod_effective, srf_wvl, rsrf)
        with wvn in cm-1, wvl in nm, fwhm in nm, wvl_central in nm, 
        SRF weighted Rayleigh optical depth, reference wavelegnth of the rsrf in nm, rsrf
    '''
    if sensor is None: 
        return 'VGT1, VGT2, Proba-V, S3A_OLCI, S3B_OLCI, S3A_SLSTR, S3B_SLSTR, '+\
               'METOP_A, METOP_B, NOAA_07, NOAA_08, NOAA_09, NOAA_10, NOAA_11, '+\
               'NOAA_12, NOAA_13, NOAA_14, NOAA_15, NOAA_16, NOAA_17, NOAA_18, NOAA_19, '+\
               'S2A_MSI, S2B_MSI, LANDSAT8_OLI, Terra_MISR'
    if (sensor=='Proba-V' and camera is None) : 
        print('{} sensor: camera needed : LEFT,RIGHT,CENTER,\ndefault CENTER'.format(sensor))
        camera='CENTER'
    import pandas as pd
    xLimits = []
    fwhm    = []
    central_wvl = []
    srf_wvl = [] 
    srf     = []

    if ('LANDSAT8' in sensor):
        platform = sensor[:8]
        fsrf  = '/rfs/proj/C3S/SRFs/OLI/LANDSAT8/Ball_BA_RSR.v1.2.xlsx'
        data   = pd.read_excel(fsrf, sheet_name='Band summary')
        bandnames = data['Band'][1:]
        for band in bandnames:
            if band=='CA' : band='CoastalAerosol'
            data = pd.read_excel(fsrf, sheet_name=band)
            srf_ = np.array(data['BA RSR [watts]'])
            srf_ = srf_/srf_.max() # normalize SRF
            ok   = srf_ > 0.005 # subset only minimum transmission
            srf_ = srf_[ok]
            srf_wvl_ = np.array(data['Wavelength'])
            srf_wvl_ = srf_wvl_[ok]
            fwhm .append(srf_wvl_[srf_>0.5][-1] - srf_wvl_[srf_>0.5][0])
            central_wvl.append((srf_wvl_[srf_>0.5][-1] + srf_wvl_[srf_>0.5][0]) * 0.5)
            xLimits.append([1e7/(srf_wvl_.max()+1.), 1e7/(srf_wvl_.min()-1.)])
            srf_wvl.append(srf_wvl_)
            srf.append(srf_)

    if 'MISR' in sensor:
        platform = sensor[:5]
        if platform=='Terra' : f='/rfs/proj/C3S/SRFs/MISR/Terra/MISR_SRF.txt'
        dat = np.loadtxt(f, skiprows=18, delimiter=',')
        nb  = dat[0,2:].size
        for i in np.arange(nb):
            srf_ = dat[:,i+2]/dat[:,i+2].max() # normalize SRF
            ok   = srf_ > 0.005 # subset only minimum transmission
            srf_ = srf_[ok]
            srf_wvl_ = dat[:,0]
            srf_wvl_ = srf_wvl_[ok]
            fwhm .append(srf_wvl_[srf_>0.5][-1] - srf_wvl_[srf_>0.5][0])
            central_wvl.append((srf_wvl_[srf_>0.5][-1] + srf_wvl_[srf_>0.5][0]) * 0.5)
            xLimits.append([1e7/(srf_wvl_.max()+1.), 1e7/(srf_wvl_.min()-1.)])
            srf_wvl.append(srf_wvl_ )
            srf.append(srf_)
            
    if 'OLCI' in sensor:
        platform = sensor[:3]
        if platform=='S3A' : fsrf=h5py.File('/rfs/proj/C3S/SRFs/OLCI/S3A/S3A_OL_SRF_20160713_mean_rsr.nc4', "r")
        else               : fsrf=h5py.File('/rfs/proj/C3S/SRFs/OLCI/S3B/S3B_OL_SRF_0_20180109_mean_rsr.nc4', "r")
        central_wvl_i = np.copy(fsrf[u'srf_centre_wavelength'])
        srf_wvl_i   = np.copy(fsrf[u"mean_spectral_response_function_wavelength"])
        srf_i       = np.copy(fsrf[u"mean_spectral_response_function"])
        fsrf.close()
        for i in np.arange(len(central_wvl_i)):
            srf_ = srf_i[i,:]/srf_i[i,:].max() # normalize SRF
            ok   = srf_ > 0.005 # subset only minimum transmission
            srf_ = srf_[ok]
            srf_wvl_ = srf_wvl_i[i,:]
            srf_wvl_ = srf_wvl_[ok]
            fwhm .append(srf_wvl_[srf_>0.5][-1] - srf_wvl_[srf_>0.5][0])
            central_wvl.append((srf_wvl_[srf_>0.5][-1] + srf_wvl_[srf_>0.5][0]) * 0.5)
            xLimits.append([1e7/(srf_wvl_.max()+1.), 1e7/(srf_wvl_.min()-1.)])
            srf_wvl.append(srf_wvl_ )
            srf.append(srf_)

    elif 'SLSTR' in sensor:
        platform = sensor[:3]
        if platform=='S3B' : fsrfs = glob('/rfs/proj/C3S/SRFs/SLSTR/S3B/SLSTR_PFM_S[123456]*.nc')
        else               : fsrfs = glob('/rfs/proj/C3S/SRFs/SLSTR/S3A/SLSTR_FM02_S[123456]*.nc')
        for f in np.sort(fsrfs):
            fsrf=h5py.File(f, "r")
            srf_wvl_     = np.copy(fsrf[u"wavelength"])
            srf_         = np.copy(fsrf[u"response"])
            srf_ /= srf_.max() # normalize SRF
            ok = srf_ > 0.005 # subset only minimum transmission
            srf_ = srf_[ok]
            srf_wvl_ = srf_wvl_[ok] * 1e3
            fwhm .append(srf_wvl_[srf_>0.5][-1] - srf_wvl_[srf_>0.5][0])
            central_wvl.append((srf_wvl_[srf_>0.5][-1] + srf_wvl_[srf_>0.5][0]) * 0.5)
            fsrf.close()
            xLimits.append([1e7/(srf_wvl_.max()+1.), 1e7/(srf_wvl_.min()-1.)])
            srf_wvl.append(srf_wvl_ )
            srf.append(srf_)

    elif ('METOP' in sensor) or ('NOAA' in sensor):
        fsrfs = glob('/rfs/proj/C3S/SRFs/AVHRR/'+sensor+'_A*.txt')           
        for f in np.sort(fsrfs):
            header       = open(f).readline()
            coef         = 1 if 'nm' in header else 1e3
            fsrf         = np.loadtxt(f, skiprows=1)
            if not 'Wavenumber' in header:
                ind          = np.argsort(fsrf[:,0])
                srf_wvl_     = fsrf[ind,0]
                srf_         = fsrf[ind,1]
            else :
                ind          = np.argsort(fsrf[:,1])
                srf_wvl_     = fsrf[ind,1]
                srf_         = fsrf[ind,2]
            srf_ /= srf_.max() # normalize SRF
            ok = srf_ > 0.005 # subset only minimum transmission
            srf_ = srf_[ok]
            srf_wvl_ = srf_wvl_[ok] * coef
            fwhm .append(srf_wvl_[srf_>0.5][-1] - srf_wvl_[srf_>0.5][0])
            central_wvl.append((srf_wvl_[srf_>0.5][-1] + srf_wvl_[srf_>0.5][0]) * 0.5)
            xLimits.append([1e7/(srf_wvl_.max()+1.), 1e7/(srf_wvl_.min()-1.)])
            srf_wvl.append(srf_wvl_ )
            srf.append(srf_)

    elif ('VGT' in sensor) or ('Proba' in sensor) :
        fsrfs  = glob('/rfs/proj/C3S/SRFs/VGT/VGT_SRF.XLSX')
        data   = pd.read_excel(fsrfs[0], sheet_name=sensor)
        if sensor=='Proba-V' : 
            data.rename(index=str, columns={"NIR  CENTER": "NIR CENTER"}, inplace=True)
            sensor2 = sensor+'-'+camera
        else : sensor2 = sensor
        for band in ['BLUE','RED','NIR','SWIR']:
            if sensor2=='Proba-V-CENTER' :
                srf_wvl_     = np.array(data['wvl_{}'.format(band)].values)
                srf_         = np.array(data['{} CENTER'.format(band)].values)
            elif sensor2=='Proba-V-LEFT' :
                srf_wvl_     = np.array(data['wvl_{}'.format(band)].values)
                srf_         = np.array(data['{} LEFT'.format(band)].values)
            elif sensor2=='Proba-V-RIGHT' :
                srf_wvl_     = np.array(data['wvl_{}'.format(band)].values)
                srf_         = np.array(data['{} RIGHT'.format(band)].values)
            elif sensor2=='VGT1' :
                srf_wvl_     = np.array(data['wavelength'].values)*1e3
                srf_         = np.array(data['{} {}'.format(band, sensor)].values)
            else :
                srf_wvl_     = np.array(data['wavelength'].values)
                srf_         = np.array(data['{} {}'.format(band, sensor)].values)    
            srf_ /= np.nanmax(srf_) # normalize SRF
            ok = srf_ > 0.005 # subset only minimum transmission
            srf_ = srf_[ok]
            srf_wvl_ = srf_wvl_[ok]
            fwhm .append(srf_wvl_[srf_>0.5][-1] - srf_wvl_[srf_>0.5][0])
            central_wvl.append((srf_wvl_[srf_>0.5][-1] + srf_wvl_[srf_>0.5][0]) * 0.5)
            xLimits.append([1e7/(srf_wvl_.max()+1.), 1e7/(srf_wvl_.min()-1.)])
            srf_wvl.append(srf_wvl_ )
            srf.append(srf_)

    elif ('MSI' in sensor):
        platform = sensor[:3]
        fsrfs  = glob('/rfs/proj/C3S/SRFs/MSI/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.0-1.xlsx')
        data   = pd.read_excel(fsrfs[0], sheet_name='Spectral Responses ({})'.format(platform))
        srf_wvl_     = np.array(data['SR_WL'])
        for b,band in enumerate(data):
            if b==0: continue
            srf_  = np.array(data[band])
            ok = srf_ > 0.0005 # subset only minimum transmission
            srf_ = srf_[ok]
            srf_wvl__ = srf_wvl_[ok]
            fwhm .append(srf_wvl__[srf_>0.5][-1] - srf_wvl__[srf_>0.5][0])
            central_wvl.append((srf_wvl__[srf_>0.5][-1] + srf_wvl__[srf_>0.5][0]) * 0.5)
            xLimits.append([1e7/(srf_wvl__.max()+1.), 1e7/(srf_wvl__.min()-1.)])
            srf_wvl.append(srf_wvl__ )
            srf.append(srf_)
            
    # wavelengths intervals
    central_wvl = np.array(central_wvl)
    fwhm = np.array(fwhm)
    srf = np.array(srf)
    srf_wvl = np.array(srf_wvl)
    ODR = []
    for w,s in zip(srf_wvl,srf):
        od = np.squeeze(rod(w*1e-3, np.array(400), 45., 0., 1013.25))
        ODR.append(simps(s*od, x=w)/simps(s,x=w))
    return np.array(xLimits), 1e7/np.array(xLimits)[:,::-1], fwhm, central_wvl, np.array(ODR), srf_wvl, srf


def date_to_float(d, epoch=np.datetime64('1980-01-01T00:00:00.000000000')):
    '''
        transform the date into a duration in minutes since epoch
        by default from '1980-01-01T00:00:00.000000000'
    '''
    
    return (d - epoch).astype(np.float64)/1.0e9/60.


def Ps(z,p0,T, g=9.801, R=287.058, lam=-0.006):
    T1 = np.log(R*T) - np.log(-R*lam*z+R*T)

    return p0*np.exp(-g/(R*lam)*T1)


def dPsdz(z,p0,T, g=9.801, R=287.058, lam=-0.006):

    return g*Ps(z,p0,T, g=9.801, R=287.058, lam=-0.006)/(R*(T-lam*z))


def pre_brdf(fbrdf):
    '''
        Read the BRDF level 3 parameters from the Albedo chain
        and compute MLUT of normalized k1p=k1/k0 and k2p=k2/k0 BRDF coefficients
    '''
    brdf = xarray.open_dataset(fbrdf)
    brdf_lut = MLUT()
    # Add the good axes
    brdf_lut.add_axis('lat',  brdf.LAT.data[::-1,0])
    brdf_lut.add_axis('lon',  brdf.LON.data[0,:])
    brdf_lut.add_axis('nband',brdf.NBAND.data)
    brdf_lut.add_axis('kernel_index',np.arange(2))
    kp12 = np.zeros((brdf_lut.axis('lat').size, brdf_lut.axis('lon').size,
                     brdf_lut.axis('nband').size, 2))
    kp12[:,:,:,0] = brdf['K012'].data[::-1,:,:,1]/brdf['K012'].data[::-1,:,:,0]
    kp12[:,:,:,1] = brdf['K012'].data[::-1,:,:,2]/brdf['K012'].data[::-1,:,:,0]
    # No BRDF good data -> 0. for Kp that means assuming lambertian surface in AC
    kp12[np.isnan(kp12)] = 0.
    brdf_lut.add_dataset('kp12', kp12, axnames=['lat','lon','nband','kernel_index'])
    brdf_lut.save('/home/did/RTC/smaccl/kp12_lut.nc', overwrite=True)

    return brdf_lut


def pre_merra2(faero, fptwo):
    '''
        Read MERRA2 2 ancillary data files and store all information into a MLUT object for further use
        (mainly for spatial and temporal interpolation)
    '''
    merra = xarray.open_dataset(faero)
    merra_lut = MLUT()
    # Add the good axes
    merra_lut.add_axis('time', date_to_float(merra.time.data)) # float array of delta time in ns from epoch time
    merra_lut.add_axis('lat',  merra.lat.data)
    merra_lut.add_axis('lon',  merra.lon.data)
    Tau = merra['TOTEXTTAU'].data
    merra_lut.add_dataset('TOTEXTTAU', Tau, axnames=['time','lat','lon'])
    merra_lut.add_dataset('TOTSCATAU', merra['TOTSCATAU'].data, axnames=['time','lat','lon'])
    merra_lut.add_dataset('TOTANGSTR', merra['TOTANGSTR'].data, axnames=['time','lat','lon'])
    merra_lut.add_dataset('BC_FRAC' , merra['BCEXTTAU'].data/Tau , axnames=['time','lat','lon'])
    merra_lut.add_dataset('DU_FRAC' , merra['DUEXTTAU'].data/Tau , axnames=['time','lat','lon'])
    merra_lut.add_dataset('OC_FRAC' , merra['OCEXTTAU'].data/Tau , axnames=['time','lat','lon'])
    merra_lut.add_dataset('SS_FRAC' , merra['SSEXTTAU'].data/Tau , axnames=['time','lat','lon'])
    merra_lut.add_dataset('SU_FRAC' , merra['SUEXTTAU'].data/Tau , axnames=['time','lat','lon'])
  
    merra = xarray.open_dataset(fptwo)
    merra_lut.add_dataset('TO3',  merra['TO3'].data,  axnames=['time','lat','lon'])
    merra_lut.add_dataset('SLP',  merra['SLP'].data,  axnames=['time','lat','lon'])
    merra_lut.add_dataset('T10M', merra['T10M'].data, axnames=['time','lat','lon'])
    merra_lut.add_dataset('TQV',  merra['TQV'].data,  axnames=['time','lat','lon'])
    del merra, Tau

    return merra_lut


def pre_aer_models(faer):
    '''
        Read MERRA2 aerosols components fraction of the aerosol models
    '''
    match = {'sulf':'SU', 'dust':'DU', 'oc':'OC', 'ssalt':'SS', 'bc':'BC'}
    f = open(faer, 'r')
    frac_aer_model = {}
    for key in match.keys():
        f.readline ()
        line = f.readline ()
        frac_aer_model[key] =  np.array(line.split()).astype(float)
    f.close()
    
    return frac_aer_model


def closest_model(X, Xb):
    '''
    return the closest model number compared to reference basis
    it is a distance minimization in a 5-dimensional space
    '''

    return np.sum((X-Xb)**2, axis=0).argmin(axis=0)


def closest_models(X, Xb):
    '''
    return the 10 closest model numbers compared to reference basis
    it is a distance minimization in a 5-dimensional space
    '''

    return np.sum((X-Xb)**2, axis=0).argsort(axis=0)[:10, :]

def load_cams(filename):
    grbs = pg.open(filename)
    datasets = {'TQV':'Total column water vapour', 'T10M':'2 metre temperature', 'TO3':'GEMS Total column ozone','TOTEXTTAU':'Total Aerosol Optical Depth at 550nm','SLP':'Mean sea level pressure'}
    ntimes = grbs.messages//len(datasets)

    data, lat, lon = grbs.read()[0].data()
    x,y = data.shape
    grbs.rewind()

    datas = {} 
    for k, n in datasets.items():
        grb= grbs.select(name=n)
        datas[k] = np.zeros((ntimes, x, y), dtype='float')
        times = []
        for i, grb in enumerate(grb):
            times.append(np.datetime64(grb.validDate+timedelta(hours=3*i)))
            data, lat, lon = grb.data()
            datas[k][i,:,:] = data[:,:]

    times = (np.array(times)-np.datetime64('1980-01-01T00:00:00.000000000')).astype('float32')/1e9/60.

    cams_lut = MLUT()
    cams_lut.add_axis('time', times)
    cams_lut.add_axis('lat', lat[:,0])
    cams_lut.add_axis('lon', lon[0])

    for k in datasets:
        cams_lut.add_dataset(k, datas[k], axnames=['time','lat','lon'])

    return cams_lut

def set_ac_flag(aot, sza, vza, climato):
    flag = np.zeros(aot.shape, dtype='int32')
    flag[(aot<=.5)] = 0
    flag[((aot>.5) & (aot<=1.))] = 2
    flag[((aot>1.) & (aot<=1.5))] = 4
    flag[(aot>1.5)] = 6

    flag[(sza>65)] |= 8
    flag[(vza>65)] |= 16

    if climato:
        flag |= 32

    return flag


def load_brdf(file_brdf):
    return np.loadtxt(file_brdf)


def F1_rtls(ths, thv, phi):   #  rossthick-lisparse, only F1
    phi[phi<0] = phi[phi<0] + 2.*np.pi
    phi[phi>np.pi] = 2.*np.pi - phi[phi>np.pi] 
    cos_xi = np.cos(ths) * np.cos(thv) + np.sin(ths) * np.sin(thv) * np.cos(phi)    
    mm     = 1./np.cos(thv) + 1./np.cos(ths)
    cos_t  = 2./mm * np.sqrt(np.tan(thv)**2 + np.tan(ths)**2 - \
             2*np.tan(thv)*np.tan(ths)*np.cos(phi) + (np.tan(thv)*np.tan(ths)*np.sin(phi))**2)
    cos_t  = np.minimum(cos_t, 1.)
    t      = np.arccos(cos_t)
    sin_t  = np.sin(t)
    big_O  = mm * (t -sin_t*cos_t)/np.pi            
    # geometric kernel
    F1     = big_O - (1./np.cos(thv) + 1./np.cos(ths)) + (1 + cos_xi)/(np.cos(thv)*np.cos(ths))/2.
    
    return F1


def F2_rtls( ths, thv, phi ): #  rossthick-lisparse, only F2
    phi[phi<0] = phi[phi<0] + 2.*np.pi
    phi[phi>np.pi] = 2.*np.pi - phi[phi>np.pi] 
    cos_xi = np.cos(ths) * np.cos(thv) + np.sin(ths) * np.sin(thv) * np.cos(phi)    
    xi     = np.arccos(cos_xi)    
    # volume-scattering kernel
    F2 = (((np.pi/2. - xi)*cos_xi + np.sin(xi))/(np.cos(thv) + np.cos(ths))) - np.pi/4.
    
    return F2
