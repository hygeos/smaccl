import numpy as np
from glob import glob
import h5py
import sys
sys.path.insert(0, '/home/did/RTC/SMART-G/')
from smartg.atmosphere import rod, simps
def SRF(sensor=None):
    '''
    Arguments:
        one sensor name in the list return by SRF()
        
    returns:
        (wvn_limits, wvl_limits, fwhm, wvl_central, rod_effective, srf_wvl, rsrf)
        with wvn in cm-1, wvl in nm, fwhm in nm, wvl_central in nm, 
        SRF weighted Rayleigh optical depth, reference wavelegnth of the rsrf in nm, rsrf
    '''
    if sensor is None: 
        return 'VGT1, VGT2, Proba-V, S3A_OLCI, S3B_OLCI, S3_SLSTR, S3B_SLSTR, '+\
               'METOP_A, METOP_B, NOAA_07, NOAA_08, NOAA_09, NOAA_10, NOAA_11,'+\
               'NOAA_12, NOAA_13, NOAA_14, NOAA_15, NOAA_16, NOAA_17, NOAA_18, NOAA_19'
    import pandas as pd
    xLimits = []
    fwhm    = []
    central_wvl = []
    srf_wvl = []
    srf     = []
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
        platform = sensor[1:3]
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
        if sensor=='Proba-V' : data.rename(index=str, columns={"NIR  CENTER": "NIR CENTER"}, inplace=True)
        for band in ['BLUE','RED','NIR','SWIR']:
            if sensor=='Proba-V' :
                srf_wvl_     = np.array(data['wvl_{}'.format(band)].values)
                srf_         = np.array(data['{} CENTER'.format(band)].values)
            elif sensor=='VGT1' :
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
