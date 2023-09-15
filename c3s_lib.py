import numpy as np
from glob import glob
import h5py
from luts.luts import MLUT
import xarray
from scipy.integrate import simps
from scipy.constants import codata

def isnumeric(x):
    try:
        float(x)
        return True
    except TypeError:
        return False

def FN2(lam):
    ''' depolarisation factor of N2
        lam : um
    '''
    return 1.034 + 3.17 *1e-4 *lam**(-2)


def FO2(lam):
    ''' depolarisation factor of O2
        lam : um
    '''
    return 1.096 + 1.385 *1e-3 *lam**(-2) + 1.448 *1e-4 *lam**(-4)

def vapor_pressure(T):
    T0=273.15
    A=T0/T
    Avogadro = codata.value('Avogadro constant')
    M_H2O=18.015
    mh2o=M_H2O/Avogadro
    return A*np.exp(18.916758 - A * (14.845878 + A*2.4918766))/mh2o/1.e6


def Fair(lam, co2):
    ''' depolarisation factor of air for CO2 (N wavelengths x M layers)
        lam : um (N)
        co2 : ppm (M)
    '''
    _FN2 = FN2(lam).reshape((-1,1))
    _FO2 = FO2(lam).reshape((-1,1))
    _CO2 = co2.reshape((1,-1))

    return ((78.084 * _FN2 + 20.946 * _FO2 + 0.934 +
            _CO2*1e-4 *1.15)/(78.084+20.946+0.934+_CO2*1e-4))


def n300(lam):
    ''' index of refraction of dry air  (300 ppm CO2)
        lam : um
    '''
    return 1e-8 * ( 8060.51 + 2480990/(132.274 - lam**(-2)) + 17455.7/(39.32957 - lam**(-2))) + 1.


def n_air(lam, co2):
    ''' index of refraction of dry air (N wavelengths x M layers)
        lam : um (N)
        co2 : ppm (M)
    '''
    N300 = n300(lam).reshape((-1,1))
    CO2 = co2.reshape((1,-1))
    return ((N300 - 1) * (1 + 0.54*(CO2*1e-6 - 0.0003)) + 1.)

def ma(co2):
    ''' molecular volume
        co2 : ppm
    '''
    return 15.0556 * co2*1e-6 + 28.9595

def raycrs(lam, co2):
    ''' Rayleigh cross section (N wavelengths x M layers)
        lam : um (N)
        co2 : ppm ((M)
    '''
    LAM = lam.reshape((-1,1))
    Avogadro = codata.value('Avogadro constant')
    Ns = Avogadro/22.4141 * 273.15/288.15 * 1e-3
    nn2 = n_air(lam, co2)**2
    return (24*np.pi**3 * (nn2-1)**2/(LAM*1e-4)**4/Ns**2/(nn2+2)**2 * Fair(lam, co2))

def g0(lat):
    ''' gravity acceleration at the ground
        lat : deg
    '''
    assert isnumeric(lat)
    return (980.6160 * (1. - 0.0026372 * np.cos(2*lat*np.pi/180.)
            + 0.0000059 * np.cos(2*lat*np.pi/180.)**2))

def g(lat, z) :
    ''' gravity acceleration at altitude z
        lat : deg (scalar)
        z : m
    '''
    assert isnumeric(lat)
    return (g0(lat) - (3.085462 * 1.e-4 + 2.27 * 1.e-7 * np.cos(2*lat*np.pi/180.)) * z
            + (7.254 * 1e-11 + 1e-13 * np.cos(2*lat*np.pi/180.)) * z**2
            - (1.517 * 1e-17 + 6 * 1e-20 * np.cos(2*lat*np.pi/180.)) * z**3)

def rod(lam, co2=400., lat=45., z=0., P=1013.25, pressure='surface'):
    """
    Rayleigh optical depth from Bodhaine et al, 99 (N wavelengths x M layers)
        lam : wavelength in um (N)
        co2 : ppm (M)
        lat : deg (scalar)
        z : altitude in m (M)
        P : pressure in hPa (M)
            (surface or sea-level)
        pressure: str
            - 'surface': P provided at altitude z
            - 'sea-level': P provided at altitude 0
    """
    Avogadro = codata.value('Avogadro constant')
    zs = 0.73737 * z + 5517.56  # effective mass-weighted altitude
    G = g(lat, zs)
    # air pressure at the pixel (i.e. at altitude) in hPa
    if pressure == 'sea-level':
        Psurf = (P * (1. - 0.0065 * z / 288.15) ** 5.255) * 1000.  # air pressure at pixel location in dyn / cm2, which is hPa * 1000
    elif pressure == 'surface':
        Psurf = P * 1000.  # convert to dyn/cm2
    else:
        raise ValueError('Invalid pressure type ({pressure})')

    return raycrs(lam, co2) * Psurf * Avogadro/ma(co2)/G

def refractivity(lam,P,T,co2):
    ''' Refractivity of air
        lam : um (N)
        P   : hPa (M)
        T   : K (M)
        co2 : ppm (M)
    '''
    p= P*100.
    t = T-273.15
    Ntp = 1 + (n_air(lam[:],co2) - 1) * p * (1.+p*(60.1-0.972*t)*1e-10)\
        /(96095.43 * (1 + 0.003661 * t))
    return Ntp



def SRF_old(sensor=None, camera=None):
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
               'S2A_MSI, S2B_MSI, LANDSAT8_OLI, Terra_MISR, EOS_1_MODIS, EOS_2_MODIS, '+\
               'JPSS_0_VIIRS, NOAA_20_VIIRS'
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
            
    elif ('MODIS' in sensor):
        fsrfs = glob('/rfs/proj/C3S/SRFs/MODIS/rtcoef_'+sensor.lower()+'_srf*.txt')        
        for f in np.sort(fsrfs):
            fsrf         = np.loadtxt(f, skiprows=4)
            srf_wvl_     = 1e7/fsrf[:,0][::-1]
            srf_         = fsrf[:,1][::-1]
            srf_ /= srf_.max() # normalize SRF
            ok = srf_ > 0.005 # subset only minimum transmission
            srf_ = srf_[ok]
            srf_wvl_ = srf_wvl_[ok] 
            fwhm .append(srf_wvl_[srf_>0.5][-1] - srf_wvl_[srf_>0.5][0])
            central_wvl.append((srf_wvl_[srf_>0.5][-1] + srf_wvl_[srf_>0.5][0]) * 0.5)
            xLimits.append([1e7/(srf_wvl_.max()+1.), 1e7/(srf_wvl_.min()-1.)])
            srf_wvl.append(srf_wvl_ )
            srf.append(srf_)
            
    elif ('VIIRS' in sensor):
        fsrfs = glob('/rfs/proj/C3S/SRFs/VIIRS/rtcoef_'+sensor.lower()+'_srf*.txt') 
        for f in np.sort(fsrfs):
            fsrf         = np.loadtxt(f, skiprows=4)
            srf_wvl_     = 1e7/fsrf[:,0][::-1]
            srf_         = fsrf[:,1][::-1]
            srf_ /= srf_.max() # normalize SRF
            ok = srf_ > 0.005 # subset only minimum transmission
            srf_ = srf_[ok]
            srf_wvl_ = srf_wvl_[ok] 
            fwhm .append(srf_wvl_[srf_>0.5][-1] - srf_wvl_[srf_>0.5][0])
            central_wvl.append((srf_wvl_[srf_>0.5][-1] + srf_wvl_[srf_>0.5][0]) * 0.5)
            xLimits.append([1e7/(srf_wvl_.max()+1.), 1e7/(srf_wvl_.min()-1.)])
            srf_wvl.append(srf_wvl_ )
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


def SRF(sensor=None, camera=None):
    '''
    Arguments:
        sensor : one sensor name in the list return by SRF()
        camera : eventually a camera name (str) for a sensor
        
    returns:
        (wvn_limits, wvl_limits, fwhm, wvl_central, rod_effective, srf_wvl, rsrf)
        with wvn in cm-1, wvl in nm, fwhm in nm, wvl_central in nm, 
        SRF weighted Rayleigh optical depth, reference wavelegnth of the rsrf in nm, rsrf, name (string)
    '''
    dir_EUMETSAT_SRFs = '/archive/proj/C3S/SRFs/EUMETSAT-SAF-SRFs/'
    dir_SRFs          = '/archive/proj/C3S/SRFs/'
    if sensor is None: 
        list_sensor_eumetsat = np.sort([f.split('/')[-1][7:-8].upper() for f in glob(dir_EUMETSAT_SRFs+'*tar')])
        list_sensor_special  = ['SENTINEL3_1_OLCI', 'SENTINEL3_2_OLCI', 'VGT1', 'VGT2', 'Proba-V',\
                                'LANDSAT_8_OLI', 'EOS_1_MISR', 'ENVISAT_MERIS']
        a=''
        for s in list_sensor_eumetsat:
            if a=='': a=a+s
            else : a=a+','+s
        for s in list_sensor_special:
            a=a+','+s
            
        return a
    
    if (sensor=='Proba-V' and camera is None) : 
        print('{} sensor: camera needed : LEFT,RIGHT,CENTER,\ndefault CENTER'.format(sensor))
        camera='CENTER'
    import pandas as pd
    xLimits = []
    fwhm    = []
    central_wvl = []
    srf_wvl = [] 
    srf     = []
    name    = []

    if 'LANDSAT' in sensor :
        platform = sensor[:8]
        fsrf   = dir_SRFs + 'OLI/LANDSAT8/Ball_BA_RSR.v1.2.xlsx'
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
            name.append(band)

    elif 'MISR' in sensor :
        platform = sensor[:5]
        f= dir_SRFs + 'MISR/Terra/MISR_SRF.txt'
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
            
    elif ('VGT' in sensor) or ('Proba' in sensor) :
        fsrfs  = glob(dir_SRFs + 'VGT/VGT_SRF.XLSX')
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
            name.append(band)
            
    elif 'OLCI' in sensor:
        platform = sensor[:3]
        if platform=='SENTINEL3_1' : fsrf=h5py.File(dir_SRFs + 'OLCI/S3A/S3A_OL_SRF_20160713_mean_rsr.nc4', "r")
        else                       : fsrf=h5py.File(dir_SRFs + 'OLCI/S3B/S3B_OL_SRF_0_20180109_mean_rsr.nc4', "r")
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
            name.append('band %i'%i)

    elif 'MERIS' in sensor:
        platform = 'ENVISAT'
        fsrfs    = pd.read_excel(dir_SRFs + 'MERIS/MERIS_NominalSRF_Model2004.xls', sheet_name='NominalSRF Model2004', skiprows=1)
        for i in range(15):
            if i==0 : st=''
            else :  st='.'+str(i)
            srf_wvl_ = np.array(fsrfs['wavelength' + st])
            srf_wvl_ = srf_wvl_[np.isfinite(srf_wvl_)]
            srf_     = np.array(fsrfs['SRF' + st])
            srf_     = srf_[np.isfinite(srf_)]
            srf_ /= srf_.max() # normalize SRF
            ok = srf_ > 0.005 # subset only minimum transmission
            srf_ = srf_[ok]
            srf_wvl_ = srf_wvl_[ok] 
            fwhm .append(srf_wvl_[srf_>0.5][-1] - srf_wvl_[srf_>0.5][0])
            central_wvl.append((srf_wvl_[srf_>0.5][-1] + srf_wvl_[srf_>0.5][0]) * 0.5)
            xLimits.append([1e7/(srf_wvl_.max()+1.), 1e7/(srf_wvl_.min()-1.)])
            srf_wvl.append(srf_wvl_ )
            srf.append(srf_)
            name.append('band %i'%i)
            
    else:
        fsrfs = glob(dir_EUMETSAT_SRFs + 'rtcoef_'+sensor.lower()+'_srf*.txt') 
        for f in np.sort(fsrfs):
            name.append(open(f,"r").readline().split()[-1])
            fsrf         = np.loadtxt(f, skiprows=4)
            srf_wvl_     = 1e7/fsrf[:,0][::-1]
            srf_         = fsrf[:,1][::-1]
            srf_ /= srf_.max() # normalize SRF
            ok = srf_ > 0.005 # subset only minimum transmission
            srf_ = srf_[ok]
            srf_wvl_ = srf_wvl_[ok] 
            fwhm .append(srf_wvl_[srf_>0.5][-1] - srf_wvl_[srf_>0.5][0])
            central_wvl.append((srf_wvl_[srf_>0.5][-1] + srf_wvl_[srf_>0.5][0]) * 0.5)
            xLimits.append([1e7/(srf_wvl_.max()+1.), 1e7/(srf_wvl_.min()-1.)])
            srf_wvl.append(srf_wvl_ )
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
    return np.array(xLimits), 1e7/np.array(xLimits)[:,::-1], fwhm, central_wvl, np.array(ODR), srf_wvl, srf, name
    
    
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
    kp12[abs(kp12)>4] = 0.
    brdf_lut.add_dataset('kp12', kp12, axnames=['lat','lon','nband','kernel_index'])

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


#def closest_model_vito(frac_aer_model, match):
def closest_model_vito(X, Xb):
    #nb_pixel = len(match['sulf'])
    nb_pixel  = X.shape[1]
    iaero = np.zeros(nb_pixel)
    #nb_model = len(frac_aer_model['sulf'])
    nb_model  = Xb.shape[1]
    xm = np.zeros((5, 1))
    #xb = np.zeros((5, nb_model))
    #xb[0, :] = frac_aer_model['sulf']
    #xb[1, :] = frac_aer_model['dust']
    #xb[2, :] = frac_aer_model['oc']
    #xb[3, :] = frac_aer_model['ssalt']
    #xb[4, :] = frac_aer_model['bc']
    for j in range(0, nb_pixel):
        xm[0] = X[0,j]
        xm[1] = X[1,j]
        xm[2] = X[2,j]
        xm[3] = X[3,j]
        xm[4] = X[4,j]
        #xm[1] = match['dust'][j]
        #xm[2] = match['oc'][j]
        #xm[3] = match['ssalt'][j]
        #xm[4] = match['bc'][j]
        iaero[j] = np.sum((np.tile(xm,nb_model) - Xb)**2,axis=0).argmin(axis=0)

    return iaero


def closest_model(X, Xb):
    '''
    return the closest model number compared to reference basis
    it is a distance minimization in a 5-dimensional space
    '''

    return np.sum((X[:, np.newaxis, :]-Xb[:, :, np.newaxis])**2, axis=0).argmin(axis=0)


def closest_models(X, Xb):
    '''
    return the 10 closest model numbers compared to reference basis
    it is a distance minimization in a 5-dimensional space
    '''

    return np.sum((X[:, np.newaxis, :]-Xb[:, :, np.newaxis])**2, axis=0).argsort(axis=0)[:10, :]


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
    if (isinstance(phi, np.ndarray)):
        phi[phi<0] = phi[phi<0] + 2.*np.pi
        phi[phi>np.pi] = 2.*np.pi - phi[phi>np.pi] 
    else:
        if phi<0 : phi+=2*np.pi
        if phi>np.pi : phi = 2*np.pi - phi
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
    if (isinstance(phi, np.ndarray)):
        phi[phi<0] = phi[phi<0] + 2.*np.pi
        phi[phi>np.pi] = 2.*np.pi - phi[phi>np.pi] 
    else:
        if phi<0 : phi+=2*np.pi
        if phi>np.pi : phi = 2*np.pi - phi
    cos_xi = np.cos(ths) * np.cos(thv) + np.sin(ths) * np.sin(thv) * np.cos(phi)    
    xi     = np.arccos(cos_xi)    
    # volume-scattering kernel
    F2 = (((np.pi/2. - xi)*cos_xi + np.sin(xi))/(np.cos(thv) + np.cos(ths))) - np.pi/4.
    
    return F2


def smaccl_dir(ca, tetas, tetav, phis, phiv, uh2o, uo3, taup550, pression, rsurf, k1p, k2p, iaero, ib):
 
    NRUN=3 # number of run necessary to compute some Jacobians with finite difference (here 2 Jacobians)   + 1 reference run 
    dpre = 10.; # absolute perturbation in pressure (hPa) for Jacobian
    dtau_rel = 0.1; # relative perturbation in AOT(550) for Jacobian
    dtau = dtau_rel * taup550;

    # loop on number of run necessary to compute some Jacobians with finite difference
    for ir in np.arange(NRUN) :
        if (ir==0): pression -= dpre
        if (ir==1): taup550  -= dtau

        us = np.cos (np.radians(tetas))
        uv = np.cos (np.radians(tetav))
        dphi = np.radians(phis-phiv)
        Peq = pression/1013.0

        ##------ 1) air mass */
        m =  1./us + 1./uv;

        ##------  2) aerosol optical depth in the spectral band, taup  */
        taup = ca[ib, iaero]['a0taup'] + ca[ib, iaero]['a1taup'] * taup550

        ##------  3) gaseous transmissions (downward and upward paths)*/
        to3 = 1.0 
        th2o= 1.0 
        to2 = 1.0 
        tco2= 1.0 
        tch4= 1.0 
        tno2= 1.0
        tco = 1.0

        uo2 = np.power (Peq , (ca[ib, iaero]['po2']))
        uco2= np.power (Peq , (ca[ib, iaero]['pco2']))
        uch4= np.power (Peq , (ca[ib, iaero]['pch4']))
        uno2= np.power (Peq , (ca[ib, iaero]['pno2']))
        uco = np.power (Peq , (ca[ib, iaero]['pco']))

        ##-----  4) if uh2o <= 0 and uo3 <= 0 no gaseous absorption is computed*/
        if( (uh2o> 0.) or ( uo3 > 0.) ) :
            to3   = np.exp ( (ca[ib, iaero]['ao3'])  * np.power ( (uo3 *m)  , (ca[ib, iaero]['no3'])  ) ) 
            th2o  = np.exp ( (ca[ib, iaero]['ah2o']) * np.power ( (uh2o*m)  , (ca[ib, iaero]['nh2o']) ) ) 
            to2   = np.exp ( (ca[ib, iaero]['ao2'])  * np.power ( (uo2 *m)  , (ca[ib, iaero]['no2'])  ) ) 
            tco2  = np.exp ( (ca[ib, iaero]['aco2']) * np.power ( (uco2*m)  , (ca[ib, iaero]['nco2']) ) ) 
            tch4  = np.exp ( (ca[ib, iaero]['ach4']) * np.power ( (uch4*m)  , (ca[ib, iaero]['nch4']) ) ) 
            tno2  = np.exp ( (ca[ib, iaero]['ano2']) * np.power ( (uno2*m)  , (ca[ib, iaero]['nno2']) ) ) 
            tco   = np.exp ( (ca[ib, iaero]['aco'])  * np.power ( (uco *m)  , (ca[ib, iaero]['nco'])  ) ) 

        ##------  5) Total scattering transmission */
        ttetas = (ca[ib, iaero]['a0T']) + (ca[ib, iaero]['a1T'])*taup550/us + ((ca[ib, iaero]['a2T'])*Peq + (ca[ib, iaero]['a3T']))/(1.+us) # /* downward */
        ttetav = (ca[ib, iaero]['a0T']) + (ca[ib, iaero]['a1T'])*taup550/uv + ((ca[ib, iaero]['a2T'])*Peq + (ca[ib, iaero]['a3T']))/(1.+uv) # /* upward   */

        ##------  6) spherical albedo of the atmosphere */
        s = (ca[ib, iaero]['a0s']) * Peq +  (ca[ib, iaero]['a3s']) + (ca[ib, iaero]['a1s'])*taup550 + (ca[ib, iaero]['a2s']) *np.power (taup550 , 2) 

        ##------  7) scattering angle cosine */
        cksi = - ( (us*uv) + (np.sqrt(1. - us*us) * np.sqrt (1. - uv*uv)*np.cos(dphi) ) )
        if isinstance(cksi, np.ndarray):
            cksi[cksi<-1] = -1.0 
        else:
            if (cksi < -1 ) : cksi=-1.0

        ##------  8) scattering angle in degree */
        ksiD = np.degrees(np.arccos(cksi))

        ##------  9) rayleigh atmospheric reflectance */
        ## pour 6s on a delta = 0.0279 */
        ray_phase = 0.7190443 * (1. + (cksi*cksi))  + 0.0412742

        taurz=(ca[ib, iaero]['taur'])*Peq

        ray_ref   = ( taurz*ray_phase ) / (4.*us*uv)

        ##-----------------Residu Rayleigh ---------*/
        Res_ray= (ca[ib, iaero]['Resr1']) + (ca[ib, iaero]['Resr2']) * taurz*ray_phase / (us*uv) + \
            (ca[ib, iaero]['Resr3']) * np.power( (taurz*ray_phase/(us*uv)),2)

        ##--------9b) Direct and scattering transmssions*/
        tautot = taup+taurz
        tdirtetas = np.exp(-tautot/us)
        tdirtetav = np.exp(-tautot/uv)
        tdiftetas = ttetas - tdirtetas
        tdiftetav = ttetav - tdirtetav

        ##------  10) aerosol atmospheric reflectance */
        aer_phase = (ca[ib, iaero]['a0P']) + (ca[ib, iaero]['a1P'])*ksiD + (ca[ib, iaero]['a2P'])*ksiD*ksiD +(ca[ib, iaero]['a3P'])*np.power(ksiD,3) + (ca[ib, iaero]['a4P']) * np.power(ksiD,4)

        ak2 = (1. - (ca[ib, iaero]['wo']))*(3. - (ca[ib, iaero]['wo'])*3*(ca[ib, iaero]['gc'])) 
        ak  = np.sqrt(ak2) 
        e   = -3.*us*us*(ca[ib, iaero]['wo']) /  (4.*(1. - ak2*us*us) ) 
        f   = -(1. - (ca[ib, iaero]['wo']))*3.*(ca[ib, iaero]['gc'])*us*us*(ca[ib, iaero]['wo']) / (4.*(1. - ak2*us*us) ) 
        dp  = e / (3.*us) + us*f 
        d   = e + f 
        b   = 2.*ak / (3. - (ca[ib, iaero]['wo'])*3*(ca[ib, iaero]['gc']))
        dell = np.exp( ak*taup )*(1. + b)*(1. + b) - np.exp(-ak*taup)*(1. - b)*(1. - b)
        ww  = (ca[ib, iaero]['wo'])/4.
        ss  = us / (1. - ak2*us*us)
        q1  = 2. + 3.*us + (1. - (ca[ib, iaero]['wo']))*3.*(ca[ib, iaero]['gc'])*us*(1. + 2.*us)
        q2  = 2. - 3.*us - (1. - (ca[ib, iaero]['wo']))*3.*(ca[ib, iaero]['gc'])*us*(1. - 2.*us)
        q3  = q2*np.exp( -taup/us )
        c1  =  ((ww*ss) / dell) * (q1*np.exp(ak*taup)*(1. + b) + q3*(1. - b) )
        c2  = -((ww*ss) / dell) * (q1*np.exp(-ak*taup)*(1. - b) + q3*(1. + b) ) 
        cp1 =  c1*ak / ( 3. - (ca[ib, iaero]['wo'])*3.*(ca[ib, iaero]['gc']) )
        cp2 = -c2*ak / ( 3. - (ca[ib, iaero]['wo'])*3.*(ca[ib, iaero]['gc']) )
        z   = d - (ca[ib, iaero]['wo'])*3.*(ca[ib, iaero]['gc'])*uv*dp + (ca[ib, iaero]['wo'])*aer_phase/4.
        x   = c1 - (ca[ib, iaero]['wo'])*3.*(ca[ib, iaero]['gc'])*uv*cp1
        y   = c2 - (ca[ib, iaero]['wo'])*3.*(ca[ib, iaero]['gc'])*uv*cp2
        aa1 = uv / (1. + ak*uv)
        aa2 = uv / (1. - ak*uv)
        aa3 = us*uv / (us + uv)

        aer_ref = x*aa1* (1. - np.exp( -taup/aa1 ) )
        aer_ref = aer_ref + y*aa2*( 1. - np.exp( -taup / aa2 )  )
        aer_ref = aer_ref + z*aa3*( 1. - np.exp( -taup / aa3 )  )
        aer_ref = aer_ref / ( us*uv )

        ##--------Residu Aerosol --------*/
        Res_aer= ( (ca[ib, iaero]['Resa1']) + (ca[ib, iaero]['Resa2']) * ( taup * m *cksi ) + (ca[ib, iaero]['Resa3']) * np.power( (taup*m*cksi ),2) ) + \
                    (ca[ib, iaero]['Resa4']) * np.power( (taup*m*cksi),3)


        ##---------Residu 6s-----------*/
        Res_6s= ( (ca[ib, iaero]['Rest1']) + (ca[ib, iaero]['Rest2']) * ( tautot * m *cksi )
            + (ca[ib, iaero]['Rest3']) * np.power( (tautot*m*cksi),2) ) + (ca[ib, iaero]['Rest4']) * np.power( (tautot*m*cksi),3)

        ##------  11) total atmospheric reflectance */
        atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s
        print(ir, aer_ref)

        ##-------- gaseous transmission */
        tg      = th2o * to3 * to2 * tco2 * tch4* tco * tno2

        ##-------- reflectance at toa*/
        ##/*------------------------*/
        ##/* Atmospheric scattering transmittance */
        ##//rsurf = rtoa - (atm_ref * tg) ;
        ax1 = F1_rtls(np.radians(tetas), np.radians(tetav), dphi)
        ax2 = F2_rtls(np.radians(tetas), np.radians(tetav), dphi)
        f1_bar_down = ca[ib, iaero]['f1d0'] + ca[ib, iaero]['f1d1']*ax1 + ca[ib, iaero]['f1d2']*ax2 
        f2_bar_down = ca[ib, iaero]['f2d0'] + ca[ib, iaero]['f2d1']*ax1 + ca[ib, iaero]['f2d2']*ax2 
        f1_bar_bar  = ca[ib, iaero]['f1b0'] + ca[ib, iaero]['f1b1']*ax1 + ca[ib, iaero]['f1b2']*ax2 
        f2_bar_bar  = ca[ib, iaero]['f2b0'] + ca[ib, iaero]['f2b1']*ax1 + ca[ib, iaero]['f2b2']*ax2
        rs          =  (1. + k1p*ax1 + k2p*ax2)
        ax1 = F1_rtls(np.radians(tetav), np.radians(tetas), np.pi-dphi)
        ax2 = F2_rtls(np.radians(tetav), np.radians(tetas), np.pi-dphi)
        f1_bar_up   = ca[ib, iaero]['f1d0'] + ca[ib, iaero]['f1d1']*ax1 + ca[ib, iaero]['f1d2']*ax2 
        f2_bar_up   = ca[ib, iaero]['f2d0'] + ca[ib, iaero]['f2d1']*ax1 + ca[ib, iaero]['f2d2']*ax2
        #/**/
        ref_surf_bar_downN= (1. + k1p*f1_bar_down + k2p*f2_bar_down)/rs
        ref_surf_bar_upN  = (1. + k1p*f1_bar_up   + k2p*f2_bar_up)  /rs
        ref_surf_bar_barN = (1. + k1p*f1_bar_bar  + k2p*f2_bar_bar )/rs
        trans_atm = (tdirtetav*tdirtetas) + \
                    (tdirtetav*tdiftetas) * (ref_surf_bar_downN) + \
                    (tdiftetav*tdirtetas) * (ref_surf_bar_upN) + \
                    (tdiftetav*tdiftetas) * (ref_surf_bar_barN)
        ##rsurf = rsurf / ( (tg * trans_atm) + (rsurf * s) ) ;
        rtoa = (rsurf * trans_atm /(1. - rsurf*s)  + atm_ref) * tg

        ## Analytical Jacobian of toa reflectance vs surface reflectance*/
        ##/*------------------------*/
        ttt   = tg * trans_atm
        rps   = s * rsurf
        nu_dir = 1./(1. - rps)
        Jr = ttt * nu_dir*nu_dir

        ## Analytical Jacobian of surface reflectance vs ozone column*/
        ##/*------------------------*/
        tgp  = tg/to3
        dtdu = (ca[ib, iaero]['ao3']*ca[ib, iaero]['no3']/uo3) * np.power ( (uo3 *m) , (ca[ib, iaero]['no3']) ) * to3
        drdt = rtoa/to3
        Juo3  = drdt * dtdu

        ## Analytical Jacobian of surface reflectance vs water vapour column*/
        ##/*------------------------*/
        tgp  = tg/th2o
        dtdu = (ca[ib, iaero]['ah2o']*ca[ib, iaero]['nh2o']/uh2o) * np.power ( (uh2o *m) , (ca[ib, iaero]['nh2o']) ) * th2o
        drdt = rtoa/th2o
        Juh2o = drdt * dtdu

        ## Finite difference Jacobians of surface reflectance vs pressure and taup550*/
        ##/*------------------------*/
        if (ir==0) : Jpre  = -rtoa
        if (ir==1) : 
            print(rtoa)
            Jtaup = -rtoa
        if (ir==2) :
            Jpre += rtoa
            Jpre /= dpre
            Jtaup += rtoa
            Jtaup /= dtau
    ## main loop (ir)

    return rtoa, Jr, Juo3, Juh2o, Jpre, Jtaup  