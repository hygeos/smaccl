import xarray
from os.path import dirname, basename
import numpy as np
from smacg import get_smac_coeffs
from glob import glob
from luts import MLUT, Idx, read_mlut
import pandas as pd
from scipy.interpolate import RectBivariateSpline

class config:
    def __init__(self):
        self.k_uh2o = 1e-1  # conversion from kg.m-2 to g.cm-2
        self.k_uo3  = 1e-3  # (for MERRA-2, Dobson) to cm.atm
        self.k_p0   = 1e-2  # pascal vers hectopascal
        self.Hatm = 8000.
        self.dir_coeff = './COEFFS/'
        self.file_dem = '/rfs/data/DEM/GTOPO30_MLUT.nc'

    def __getitem__(self, name):
        return self.__dict__[name]

def index_to_datetime(panda_dataframe):                                                                                                                                             
    time = panda_dataframe['Time(hh:mm:ss)'].values.astype('str')
    keydate = 'Date(dd-mm-yyyy)'
    if not (keydate in panda_dataframe.keys()):
        keydate = 'Date(dd-mm-yy)'
    date = panda_dataframe[keydate].values.astype('str')
#    date = panda_dataframe['Date(dd-mm-yyyy)'].values.astype('str')
    date_time = np.empty(time.size).astype('str')

    for i in range(date.size):
        date_time[i] = pd.to_datetime(str(date[i]) + ' ' + str(time[i]), format='%d:%m:%Y %H:%M:%S')
        # date_time[i] = datetime_to_timestamp(pd.to_datetime(str(date[i])+' '+str(time[i]),format='%d:%m:%Y %H:%M:%S'))

    final_dataframe = panda_dataframe
    # final_dataframe.sort_index().reindex(newIndex.sort_values(), method='ffill')

    final_dataframe.index = date_time.astype('datetime64[ns]')
    # final_dataframe.sort_index()
    # final_dataframe.index = final_dataframe.sort_index()
    del final_dataframe['Time(hh:mm:ss)']
#    del final_dataframe['Date(dd-mm-yyyy)']
    del final_dataframe[keydate]

    return final_dataframe

class read_file:
    def __init__(self, sat, filename, cfg):
        self.satellite = sat
        self.filename = filename
        self.preprocessing = {"AVHRR":self.pre_avhrr, "VGT2":self.pre_vgt}
        self.cfg = cfg
#        self.preprocessing = {"AVHRR":self.pre_avhrr}

    def close(self):
        self.filename = None

    def __enter__(self):
        self.dataset = xarray.open_dataset(self.filename)
        print self.dataset
        exit(0)

        self.preprocessing[self.satellite]()

        return self

    def __exit__(self, *args):
        self.filename = None

    def scaling(self, data, scale, offset):
        return data*scale + offset

    def pre_avhrr(self):
        self.date  = basename(dirname(self.filename))[19:27]
#        dir_coef_name = './' # directory containing SMAC Coefficients
        self.bands = ['ch1', 'ch2', 'ch3a'] #
        sensor   = 'NOAA14'  # Effective sensor 
        aer_coef = 'CONT'  # 'CONT' or 'DES' SMAC aerosol coefficient
        tab_band_coef = {'ch1':'VIS', 'ch2':'NIR', 'ch3a':'NIR'}
#        self.bands_path = ["".join((dir_coef_name, 'COEFFS/coef_' + sensor + tab_band_coef[x] + '_' + aer_coef + '.dat')) for x in self.bands]
        self.bands_path = ["".join((cfg.dir_coeff+'/coef_' + sensor + tab_band_coef[x] + '_' + aer_coef + '.dat')) for x in self.bands]

        conv = {'sun_zen':'SZA', 'sun_azi':'SAA', 'sat_zen':'VZA', 'sat_azi':'VAA'}
        self.dataset.rename(conv, inplace=True)

        for b in self.bands:
            self.dataset[b] = self.scaling(self.dataset[b], 1./100., 0.) #TODO : quand les fichiers avhrr auront leurs attributs, changer les valeurs en dure.

        localtime = basename(dirname(self.filename))[27:31]
        self.dataset['mean-time-dec'] = float(localtime[:2]) + float(localtime[2:4])/60.

    def pre_vgt(self):
        self.date  = basename(dirname(self.filename))
#        dir_coef_name = './' # directory containing SMAC Coefficients
        self.bands = ['B0', 'B2', 'B3', 'MIR']
        aer_coef = 'CONT'
        sensor = 'VGT2_'
        self.bands_path = ["".join((cfg.dir_coeff+'/coef_' + sensor + x + '_' + aer_coef + '.dat')) for x in self.bands]
        lon0, lat0, d_lon, d_lat = [float(x) for x in self.dataset.map_info.split(',')[3:7]]
        (XSIZE, YSIZE) = self.dataset['B0'].shape

        lat=lat0 - np.arange(XSIZE)*d_lat
        lon=lon0 + np.arange(YSIZE)*d_lon
        lat,lon=np.meshgrid(lat,lon)
        self.dataset['lat']=(('x', 'y'), lat)
        self.dataset['lon']=(('x', 'y'), lon)

        # preprocessing of VGT file to fill lat and lon to XSIZE and YSIZE
        # This maybe unnecessary in the future if lat and lon are given for each pixel of VGT
        self.vgt_tie_complete(self.dataset,'SZA')
        self.vgt_tie_complete(self.dataset,'SAA')
        self.vgt_tie_complete(self.dataset,'VZA')
        self.vgt_tie_complete(self.dataset,'VAA')

        for b in self.bands:
            self.dataset[b] = self.dataset[b]* float(self.dataset[b].attrs['Scale']) + float(self.dataset[b].attrs['Offset'])

        self.dataset['mean-time-dec']= 9.5

    def getData(self):
        # start with tie points
        tetas       = self.dataset['SZA'].data.astype(np.float32, order='C')
        tetav       = self.dataset['VZA'].data.astype(np.float32, order='C')
        phis        = self.dataset['SAA'].data.astype(np.float32, order='C')
        phiv        = self.dataset['VAA'].data.astype(np.float32, order='C')

        # prepare radiometry array
        NB = len(self.bands)
        self.sizes = (XSIZE, YSIZE) = self.dataset[self.bands[0]].shape
        rtoa    = np.zeros((NB, YSIZE, XSIZE), dtype='float32', order='C')

        for iband, band in enumerate(self.bands):
            rtoa[iband,:,:] = self.dataset[band].data

        # Getting SMAC coefficients
        coeffs     = get_smac_coeffs(self.bands_path)

        return tetas, tetav, phis, phiv, rtoa, coeffs

    def vgt_tie_complete (self, dataset,a):
        machin = dataset[a] * float(dataset[a].attrs['Scale']) + float(dataset[a].attrs['Offset'])
        (XSIZE, YSIZE) = self.dataset[self.bands[0]].shape
        XTIE = YTIE = 7
        r1=np.arange(0,XSIZE,XTIE)
        r11,_=np.meshgrid(np.arange(XSIZE),np.arange(YSIZE))
        r2=np.arange(0,YSIZE,YTIE)
        _,r22=np.meshgrid(np.arange(XSIZE),np.arange(YSIZE))
        res = RectBivariateSpline(r1,r2,machin).ev(r11,r22)
        res = xarray.DataArray(res,dims = ('phony_dim_0', 'phony_dim_0'))
                                    
        dataset.drop(a)
        dataset[a] = res

#def readMerra(pfile, date, k_uh2o, k_uo3, k_p0, Hatm, dem_lut=None):
def readMerra(pfile, date, cfg, dem_lut=None):
    # read merra2
    merra_lut = MLUT()

    # Add the good axis
#    year  = basename(dirname(filename))[19:23]
    year = date[:4]
    filenameMerra2Aerosol = cfg.dir_aerosol+year+'/MERRA2_*.tavg1_2d_aer_Nx.'+date+'.SUB.nc4'
    filenameMerra2Aerosol = glob(filenameMerra2Aerosol)
    if len(filenameMerra2Aerosol) != 1:
        raise Exception("error on filename merra aerosol")
    filenameMerra2Aerosol = filenameMerra2Aerosol[0]
    print(filenameMerra2Aerosol)
    merra = xarray.open_dataset(filenameMerra2Aerosol)
    merra_lut.add_axis('time', np.linspace(0.5, 23.5, num=24))
    merra_lut.add_axis('lat',  merra.lat.data)
    merra_lut.add_axis('lon',  merra.lon.data)
    sds=['BCEXTTAU','DUEXTTAU','DUEXTT25','OCEXTTAU','SSEXTT25','SSEXTTAU','SUEXTTAU','TOTANGSTR','TOTEXTTAU','TOTSCATAU']
    for sd in sds:
        merra_lut.add_dataset(sd, merra[sd].data, axnames=['time','lat','lon'])

    filenameMerra2Ozone = cfg.dir_ozone+year+'/MERRA2_*.tavg1_2d_chm_Nx.'+date+'.SUB.nc4'
    filenameMerra2Ozone = glob(filenameMerra2Ozone)
    if len(filenameMerra2Ozone) != 1:
        raise Exception('error on filename merra ozone')
    filenameMerra2Ozone = filenameMerra2Ozone[0]
    merra = xarray.open_dataset(filenameMerra2Ozone)
    sds=['TO3']
    for sd in sds:
            merra_lut.add_dataset(sd, merra[sd].data, axnames=['time','lat','lon'])

    filenameMerra2Pressure = cfg.dir_pressure+year+'/MERRA2_*.tavg1_2d_slv_Nx.'+date+'.SUB.nc4'
    filenameMerra2Pressure = glob(filenameMerra2Pressure)
    if len(filenameMerra2Pressure) != 1:
        raise Exception('error on filename merra pressure')
    filenameMerra2Pressure = filenameMerra2Pressure[0]
    merra = xarray.open_dataset(filenameMerra2Pressure)
    sds=['SLP']
    for sd in sds:
            merra_lut.add_dataset(sd, merra[sd].data, axnames=['time','lat','lon'])

    filenameMerra2Water = cfg.dir_water_vapor+year+'/MERRA2_*.inst1_2d_int_Nx.'+date+'.SUB.nc4'
    filenameMerra2Water = glob(filenameMerra2Water)
    if len(filenameMerra2Water) != 1:
        raise Exception('error on filename merra water vapor')
    filenameMerra2Water = filenameMerra2Water[0]
    merra = xarray.open_dataset(filenameMerra2Water)
    sds=['TQV']
    for sd in sds:
        merra_lut.add_dataset(sd, merra[sd].data, axnames=['time','lat','lon'])

    #interpolate merra 2 data and dem to the image location and time
    taup550 =merra_lut['TOTEXTTAU'][Idx(pfile.dataset['mean-time-dec'].data, round=False),Idx(pfile.dataset['lat'].data,round=False),Idx(pfile.dataset['lon'].data,round=False)].astype(np.float32, order='C') 

    uh2o = merra_lut['TQV'][Idx(pfile.dataset['mean-time-dec'].data, round=False),Idx(pfile.dataset['lat'].data,round=False),Idx(pfile.dataset['lon'].data,round=False)].astype(np.float32, order='C')
    # conversion from kg.m-2 to g.cm-2
    uh2o *= cfg.k_uh2o

    uo3 = merra_lut['TO3'][Idx(pfile.dataset['mean-time-dec'].data, round=False),Idx(pfile.dataset['lat'].data,round=False),Idx(pfile.dataset['lon'].data,round=False)].astype(np.float32, order='C')
    # conversion from Dobson to cm.atm 
    uo3  *= cfg.k_uo3

    pressure = merra_lut['SLP'][Idx(pfile.dataset['mean-time-dec'].data, round=False),Idx(pfile.dataset['lat'].data,round=False),Idx(pfile.dataset['lon'].data,round=False)].astype(np.float32, order='C')
#            alt = np.array((dem_lut['elev']*1.)[Idx(pfile.dataset['lat'].data,round=False),Idx(pfile.dataset['lon'].data,round=False)]).astype(np.float32, order='C')
    alt = np.array(dem_lut['elev'][Idx(pfile.dataset['lat'].data,round=False),Idx(pfile.dataset['lon'].data,round=False)]).astype(np.float32, order='C')
    # pressure correction for surface altitude and transformation from Pa to hPa
    pressure *= np.exp(-alt/cfg.Hatm) * cfg.k_p0

    return taup550, uh2o, uo3, pressure

def readAeronet(site_name, date):
    AODversion = '15'
    AVG = '10'
#    csv_url = 'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?site=' + site_name + '&year=' + year1 + '&month=' + month1 + '&day=' + day1 + '&year2=' + year2 + '&m    onth2=' + month2 + '&day2=' + day2 + '&AOD' + AODversion + '=1&AVG=' + AVG + '&if_no_html=1'
    year = date[:4]
    month = date[4:6]
    day = date[6:8]
#    names_col = np.array(['AERONET_Site', 'Date(dd-mm-yyyy)', 'Time(hh:mm:ss)', 'Day_of_Year', 
#                           'Day_of_Year(Fraction)', 'AOD_1640nm', 'AOD_1020nm', 'AOD_870nm',
#                           'AOD_865nm', 'AOD_779nm', 'AOD_675nm', 'AOD_667nm', 'AOD_620nm',
#                           'AOD_560nm', 'AOD_555nm', 'AOD_551nm', 'AOD_532nm', 'AOD_531nm',
#                           'AOD_510nm', 'AOD_500nm', 'AOD_490nm', 'AOD_443nm', 'AOD_440nm',
#                           'AOD_412nm', 'AOD_400nm', 'AOD_380nm', 'AOD_340nm',
#                           'Precipitable_Water(cm)', 'AOD_Empty', 'AOD_Empty.1', 'AOD_Empty.2',
#                           'AOD_Empty.3', 'AOD_Empty.4', 'AOD_Empty.5', 'AOD_Empty.6',
#                           'Triplet_Variability_1640', 'Triplet_Variability_1020',
#                           'Triplet_Variability_870', 'Triplet_Variability_865',
#                           'Triplet_Variability_779', 'Triplet_Variability_675',
#                           'Triplet_Variability_667', 'Triplet_Variability_620',
#                           'Triplet_Variability_560', 'Triplet_Variability_555',
#                           'Triplet_Variability_551', 'Triplet_Variability_532',
#                           'Triplet_Variability_531', 'Triplet_Variability_510',
#                           'Triplet_Variability_500', 'Triplet_Variability_490',
#                           'Triplet_Variability_443', 'Triplet_Variability_440',
#                           'Triplet_Variability_412', 'Triplet_Variability_400',
#                           'Triplet_Variability_380', 'Triplet_Variability_340',
#                           'Triplet_Variability_Precipitable_Water(cm)',
#                           'Triplet_Variability_AOD_Empty', 'Triplet_Variability_AOD_Empty.1',
#                           'Triplet_Variability_AOD_Empty.2',
#                           'Triplet_Variability_AOD_Empty.3',
#                           'Triplet_Variability_AOD_Empty.4',
#                           'Triplet_Variability_AOD_Empty.5',
#                           'Triplet_Variability_AOD_Empty.6', '440-870_Angstrom_Exponent',
#                           '380-500_Angstrom_Exponent', '440-675_Angstrom_Exponent',
#                           '500-870_Angstrom_Exponent', '340-440_Angstrom_Exponent',
#                           '440-675_Angstrom_Exponent[Polar]', 'Data_Quality_Level',
#                           'AERONET_Instrument_Number', 'Site_Latitude(Degrees)',
#                           'Site_Longitude(Degrees)', 'Site_Elevation(m)',
#                           'Solar_Zenith_Angle(Degrees)', 'Optical_Air_Mass',
#                           'Sensor_Temperature(Degrees_C)', 'Ozone(Dobson)', 'NO2(Dobson)'], dtype=object)
    csv_url = 'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?site=' + site_name + '&year=' + year + '&month=' + month + '&day=' + day + '&AOD' + AODversion + '=1&AVG=' + AVG + '&if_no_html=1'
#    csv_data = pd.read_csv(csv_url, skiprows=6, error_bad_lines=False, header=None, names=names_col, usecols=np.arange(81))
    csv_data = pd.read_csv(csv_url, error_bad_lines=False, header=5) #, names=names_col, usecols=np.arange(len(names_col)))
    csv_data = index_to_datetime(csv_data)
    # TODO: pour calculer aot550 utiliser aot440 ou aot500 ?
    aot440 = csv_data.AOD_440nm.values
    angstrom = csv_data['440-675_Angstrom_Exponent'].values
    aot550 = ((550 / 440) ** (-angstrom)) * aot440

    lat = csv_data['Site_Latitude(Degrees)'].values[0]
    lat = np.repeat(lat, 49)
    lon = csv_data['Site_Longitude(Degrees)'].values[0]
    lon = np.repeat(lon, 49)
    wv = csv_data['Precipitable_Water(cm)']
    wv_reshaped = np.repeat(wv, 49 * 49).reshape(wv.size, 49, 49)
    oz = csv_data['Ozone(Dobson)']
    oz_reshaped = np.repeat(oz, 49 * 49).reshape(oz.size, 49, 49)
    aot550_reshaped = np.repeat(aot550, 49 * 49).reshape(aot550.size, 49, 49)

    return aot550_reshaped, oz_reshaped, wv_reshaped


#def readData(listFilename, k_uh2o, k_uo3, k_p0, Hatm, dem_lut=None, sat='AVHRR'):
def readData(listFilename, cfg, dem_lut=None, sat='AVHRR'):
    iz = 0
    gpuTetav = None
    if dem_lut is None:
        fdem = cfg.file_dem
#        fdem = '/rfs/data/DEM/GTOPO30_MLUT.nc'
        dem_lut = read_mlut(fdem)

    for filename in listFilename:
        # read avhrr
        with read_file(sat, filename, cfg) as pfile:
            tetas, tetav, phis, phiv, rtoa, coeffs = pfile.getData()
            if gpuTetav is None:
                gpuTetav = np.zeros((len(listFilename), pfile.sizes[0], pfile.sizes[1]), dtype='float32')
                gpuTetas = np.zeros((len(listFilename), pfile.sizes[0], pfile.sizes[1]), dtype='float32')
                gpuPhis = np.zeros((len(listFilename), pfile.sizes[0], pfile.sizes[1]), dtype='float32')
                gpuPhiv = np.zeros((len(listFilename), pfile.sizes[0], pfile.sizes[1]), dtype='float32')
                gpuRtoa = np.zeros((len(listFilename), len(pfile.bands), pfile.sizes[0], pfile.sizes[1]), dtype='float32')
                gpuTaup550 = np.zeros((len(listFilename), pfile.sizes[0], pfile.sizes[1]), dtype='float32')
                gpuUH2O = np.zeros((len(listFilename), pfile.sizes[0], pfile.sizes[1]), dtype='float32')
                gpuUO3 = np.zeros((len(listFilename), pfile.sizes[0], pfile.sizes[1]), dtype='float32')
                gpuPressure = np.zeros((len(listFilename), pfile.sizes[0], pfile.sizes[1]), dtype='float32')
            gpuTetav[iz] = tetav
            gpuTetas[iz] = tetas
            gpuPhis[iz] = phis
            gpuPhiv[iz] = phiv
            gpuRtoa[iz] = rtoa


#            taup550, uh2o, uo3, pressure = readMerra(pfile, pfile.date, k_uh2o, k_uo3, k_p0, Hatm, dem_lut)
            taup550, uh2o, uo3, pressure = readMerra(pfile, pfile.date, cfg, dem_lut)
#            readAeronet('Hamburg', date)
#            exit(0)

            gpuTaup550[iz] = taup550
            gpuUH2O[iz] = uh2o
            gpuUO3[iz] = uo3
            gpuPressure[iz] = pressure

        iz+=1
    gpuRtoa = np.swapaxes(gpuRtoa, 1, 0)

    return coeffs, gpuTetas, gpuTetav, gpuPhis, gpuPhiv, gpuUH2O, gpuUO3, gpuTaup550, gpuPressure, gpuRtoa


if __name__=='__main__':
    filename = '/rfs/data/AVHRR/1996/199601/C3S-L1B-AVHRR_NOAA-19960101100544-fv0001.nc/testdata_ATHENS-NOA.nc'
    filename = '/rfs/data/C3S/VGT/EXTRACT/2001/20010102/0_La_Crau_V120010102066.h5'
#    with read_file('AVHRR', filename) as pfile:
#        print("ouvert")
#        print(pfile.dataset)
#        print(pfile.getData())
    k_uh2o = 1e-1  # conversion from kg.m-2 to g.cm-2
    k_uo3  = 1e-3  # (for MERRA-2, Dobson) to cm.atm
    k_p0   = 1e-2  # pascal vers hectopascal
    Hatm = 8000.

    cfg = config()
    cfg.dir_aerosol = '/rfs/data/MERRA2/aer_extinction/'
    cfg.dir_ozone = '/rfs/data/MERRA2/ozone/'
    cfg.dir_pressure = '/rfs/data/MERRA2/surf_pression/'
    cfg.dir_water_vapor = '/rfs/data/MERRA2/total_precipitable_water_vapor/'

    listFilename = [filename, filename]
#    coeffs, gpuTetas, gpuTetav, gpuPhis, gpuPhiv, gpuUH2O, gpuUO3, gpuTaup550, gpuPressure, gpuRtoa = readData(listFilename, k_uh2o, k_uo3, k_p0, Hatm, sat='VGT2')
    coeffs, gpuTetas, gpuTetav, gpuPhis, gpuPhiv, gpuUH2O, gpuUO3, gpuTaup550, gpuPressure, gpuRtoa = readData(listFilename, cfg, sat='AVHRR')
    print(gpuRtoa.shape)
#    print gpuTetas[0], gpuTetas.shape
#   print(gpuRtoa[2,0,:,:], gpuRtoa[2,1,:,:])
    print(np.unique(gpuTetas), np.unique(gpuTetav), np.unique(gpuPhis), np.unique(gpuPhiv))
