from glob import glob
from tools.luts import MLUT, Idx#, read_mlut
import xarray
import numpy as np
from read_file import index_to_datetime
import pandas as pd
import datetime as dt
from os.path import exists

def comp_aot550(data, year_end):
    filtre_date = (data.index < dt.datetime(year_end+1, 1, 1)) 
    ndata = np.sum(filtre_date)

    aot550 =            np.zeros((ndata)) + np.NaN

    aot550_440 = ((550./440.)**(-data['440-675_Angstrom_Exponent']))*data.AOD_440nm
    filtre = ((~np.isnan(aot550_440)) & (~np.isnan(data['440-675_Angstrom_Exponent'])))
    aot550[filtre] = aot550_440[filtre]

    aot550_500 = ((550./500.)**(-data['500-870_Angstrom_Exponent']))*data.AOD_500nm
    filtre = ((~np.isnan(aot550_500)) & (~np.isnan(data['500-870_Angstrom_Exponent'])))
    aot550[filtre] = aot550_500[filtre]

    filtre = (~np.isnan(data.AOD_551nm))
    aot550[filtre] = data.AOD_551nm[filtre]

    return aot550

def meanAeronet(data, hour, delta):
    dates_aeronet = np.unique(data.index.date)
    aot440 = []
    aot500 = []
    aot551 = []
    angstrom440_675 = []
    angstrom500_870 = []
    dates = []
    for date in dates_aeronet:
        date_min = dt.datetime(date.year, date.month, date.day, hour, 0, 0) - dt.timedelta(minutes=delta)
        date_max = dt.datetime(date.year, date.month, date.day, hour, 0, 0) + dt.timedelta(minutes=delta)

        a440 = data.AOD_440nm[date_min:date_max]
        a440[(a440==-999)] = np.NaN
        aot440.append(np.mean(a440))

        a500 = data.AOD_500nm[date_min:date_max]
        a500[(a500==-999)] = np.NaN
        aot500.append(np.mean(a500))

        a551 = data.AOD_551nm[date_min:date_max]
        a551[(a551==-999)] = np.NaN
        aot551.append(np.mean(a551))

        a440_675 = data['440-675_Angstrom_Exponent'][date_min:date_max]
        a440_675[(a440_675==-999)] = np.NaN
        angstrom440_675.append(np.mean(a440_675))

        a500_870 = data['500-870_Angstrom_Exponent'][date_min:date_max]
        a500_870[(a500_870==-999)] = np.NaN
        angstrom500_870.append(np.mean(a500_870))

        dates.append(dt.datetime(date.year, date.month, date.day, hour, 0, 0))

    date_pandas = pd.DatetimeIndex(dates)
    data_pandas = pd.DataFrame(data={'AOD_440nm':np.array(aot440), 'AOD_500nm':np.array(aot500), 'AOD_551nm':np.array(aot551), '440-675_Angstrom_Exponent':np.array(angstrom440_675), '500-870_Angstrom_Exponent':angstrom500_870}, index=date_pandas)

    return data_pandas

def readMerra(filename):
    merra_lut = MLUT()
    print(filename)
    merra = xarray.open_dataset(filename)

    merra_lut.add_axis('time', np.linspace(0.5, 23.5, num=24))
    merra_lut.add_axis('lat',  merra.lat.data)
    merra_lut.add_axis('lon',  merra.lon.data)

    merra_lut.add_dataset('TOTEXTTAU', merra['TOTEXTTAU'].data, axnames=['time','lat','lon'])

    return merra_lut

def readAeronet(site_name, year_start, year_end, times):
    AODversion = '15'
    AVG = '20' #daily average
    AVG = '10' #all points

    year1 = str(year_start)
    month1 = '01'
    day1 = '01'
    year2 = str(year_end)
    month2 = '12'
    day2 = '31'
    if AVG=='10':
        names_col = np.array(['AERONET_Site', 'Date(dd-mm-yyyy)', 'Time(hh:mm:ss)', 'Day_of_Year',
            'Day_of_Year(Fraction)', 'AOD_1640nm', 'AOD_1020nm', 'AOD_870nm',
            'AOD_865nm', 'AOD_779nm', 'AOD_675nm', 'AOD_667nm', 'AOD_620nm',
            'AOD_560nm', 'AOD_555nm', 'AOD_551nm', 'AOD_532nm', 'AOD_531nm',
            'AOD_510nm', 'AOD_500nm', 'AOD_490nm', 'AOD_443nm', 'AOD_440nm',
            'AOD_412nm', 'AOD_400nm', 'AOD_380nm', 'AOD_340nm',
            'Precipitable_Water(cm)', 'AOD_Empty', 'AOD_Empty.1', 'AOD_Empty.2',
            'AOD_Empty.3', 'AOD_Empty.4', 'AOD_Empty.5', 'AOD_Empty.6',
            'Triplet_Variability_1640', 'Triplet_Variability_1020',
            'Triplet_Variability_870', 'Triplet_Variability_865',
            'Triplet_Variability_779', 'Triplet_Variability_675',
            'Triplet_Variability_667', 'Triplet_Variability_620',
            'Triplet_Variability_560', 'Triplet_Variability_555',
            'Triplet_Variability_551', 'Triplet_Variability_532',
            'Triplet_Variability_531', 'Triplet_Variability_510',
            'Triplet_Variability_500', 'Triplet_Variability_490',
            'Triplet_Variability_443', 'Triplet_Variability_440',
            'Triplet_Variability_412', 'Triplet_Variability_400',
            'Triplet_Variability_380', 'Triplet_Variability_340',
            'Triplet_Variability_Precipitable_Water(cm)',
            'Triplet_Variability_AOD_Empty', 'Triplet_Variability_AOD_Empty.1',
            'Triplet_Variability_AOD_Empty.2',
            'Triplet_Variability_AOD_Empty.3',
            'Triplet_Variability_AOD_Empty.4',
            'Triplet_Variability_AOD_Empty.5',
            'Triplet_Variability_AOD_Empty.6', '440-870_Angstrom_Exponent',
            '380-500_Angstrom_Exponent', '440-675_Angstrom_Exponent',
            '500-870_Angstrom_Exponent', '340-440_Angstrom_Exponent',
            '440-675_Angstrom_Exponent[Polar]', 'Data_Quality_Level',
            'AERONET_Instrument_Number', 'Site_Latitude(Degrees)',
            'Site_Longitude(Degrees)', 'Site_Elevation(m)',
            'Solar_Zenith_Angle(Degrees)', 'Optical_Air_Mass',
            'Sensor_Temperature(Degrees_C)', 'Ozone(Dobson)', 'NO2(Dobson)'], dtype=object)

    csv_url = 'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?site=' + site_name + '&year=' + year1 + '&month=' + month1 + '&day=' + day1 + '&year2=' + year2 + '&month2=' + month2 + '&day2=' + day2 + '&AOD' + AODversion + '=1&AVG=' + AVG + '&if_no_html=1'
#    print csv_url
    try:
        if AVG=='20':
            times = [12]
            csv_data = pd.read_csv(csv_url, error_bad_lines=False, header=5) #, names=names_col, usecols=np.arange(len(names_col)))
#            aeronet_data = {12:csv_data}
        else:
#            times = [9,12,15]
            csv_data = pd.read_csv(csv_url, error_bad_lines=False, header=None, skiprows=6, names=names_col, usecols=np.arange(len(names_col)))
            csv_data = index_to_datetime(csv_data)
            aeronet_data = []
            for time in times:
                aeronet_data.append( meanAeronet(csv_data, time, 30))
#            aeronet_data_9 = meanAeronet(csv_data, 9, 30)
#            aeronet_data_12 = meanAeronet(csv_data, 12, 30)
#            aeronet_data_15 = meanAeronet(csv_data, 15, 30)
#            frames = [aeronet_data_9, aeronet_data_12, aeronet_data_15]
#            csv_data = pd.concat(frames)
            csv_data = pd.concat(aeronet_data)

    except pd.errors.ParserError:
        return None, 0


    filtre_date = (csv_data.index < dt.datetime(year_end+1, 1, 1)) 
    aot440 = csv_data.AOD_440nm.values[filtre_date]
    aot500 = csv_data.AOD_500nm.values[filtre_date]
    aot551 = csv_data.AOD_551nm.values[filtre_date]
    angstrom440_675 = csv_data['440-675_Angstrom_Exponent'].values[filtre_date]
    angstrom500_870 = csv_data['500-870_Angstrom_Exponent'].values[filtre_date]
    aot550 = comp_aot550(csv_data, year_end)
#    ndata = np.sum(filtre_date)
#
#    aot550 =            np.zeros((ndata)) + np.NaN
#    aot550_440 =        np.zeros((ndata)) + np.NaN
#    aot440 =            np.zeros((ndata)) + np.NaN
#    angstrom440_675 =   np.zeros((ndata)) + np.NaN
#    aot550_500 =        np.zeros((ndata)) + np.NaN
#    aot500 =            np.zeros((ndata)) + np.NaN
#    angstrom500_870 =   np.zeros((ndata)) + np.NaN
#    aot551 =            np.zeros((ndata)) + np.NaN
#
#    if 'AOD_440nm' in csv_data.keys():
#        aot440 = csv_data.AOD_440nm.values[filtre_date]
#        angstrom440_675 = csv_data['440-675_Angstrom_Exponent'].values[filtre_date]
#        aot550_440 = ((550./440.)**(-angstrom440_675))*aot440
#        filtre = ((~np.isnan(aot550_440)) & (aot440 != -999) & (angstrom440_675 != -999))
#        aot550[filtre] = aot550_440[filtre]
#    if 'AOD_500nm' in csv_data.keys():
#        aot500 = csv_data.AOD_500nm.values[filtre_date]
#        angstrom500_870 = csv_data['500-870_Angstrom_Exponent'].values[filtre_date]
#        aot550_500 = ((550./500.)**(-angstrom500_870))*aot500
#        filtre = ((~np.isnan(aot550_500)) & (aot500 != -999) & (angstrom500_870 != -999))
#        aot550[filtre] = aot550_500[filtre]
#    if 'AOD_551nm' in csv_data.keys():
#        aot551 = csv_data.AOD_551nm.values[filtre_date]
#        filtre = ((~np.isnan(aot551)) & (aot551 != -999))
#        aot550[filtre] = aot551[filtre]

#    data = {'aot550_500':aot550_500, 'aot550_440':aot550_440, 'aot500':csv_data.AOD_500nm.values[filtre_date], 'aot440':csv_data.AOD_440nm.values[filtre_date]}
#    aot550_500 = ((550./500.)**(-csv_data['500-870_Angstrom_Exponent'].values[filtre_date]))*csv_data.AOD_500nm.values[filtre_date]
#    aot551 = csv_data.AOD_551nm.values[filtre_date]
#    data = {'aot550': aot550, 'aot551': aot551, 'aot500':csv_data.AOD_500nm.values[filtre_date], 'aot440':csv_data.AOD_440nm.values[filtre_date], 'angstrom_400_675':csv_data['440-675_Angstrom_Exponent'].values[filtre_date], 'angstrom_500_870':csv_data['500-870_Angstrom_Exponent'].values[filtre_date]}
    data = {'aot550': aot550, 'aot551': aot551, 'aot500':aot500, 'aot440':aot440, 'angstrom_440_675':angstrom440_675, 'angstrom_500_870':angstrom500_870}

    return pd.DataFrame(data=data, index=csv_data.index[filtre_date]), csv_data.shape[0]
#    return pd.DataFrame(data=aeronet_data, index=csv_data.index[filtre_date]), csv_data.shape[0]

if __name__=="__main__":
#    year_start = 1999
#    year_end = 2005
    year_start = 1996
    year_start = 2000
    year_start = 1998
    year_start = 2002
    year_start = 1999
    year_start = 2003
    year_end = year_start
    print "date : ", year_start

#    times = [9, 12, 15]
    times = range(24)

#    cols = {'date':[], 'site':[], 'lat':[], 'lon':[], 'aeronet_taua':[], 'merra_taua':[]}
#    taua_data = pd.DataFrame(data=cols)
#    print(taua_data)


    sites_file='/rfs/proj/C3S/aeronet_locations.txt'
    sites = pd.read_csv(sites_file, skiprows=1)
#    filtre_test = (sites['Site_Name'] == 'Venise')

#    sites_filtre='/home/bruno/Projets/smacg/site_v1_1999_2005.txt'
#    sites_filtre='skip'
#    if exists(sites_filtre):
#        sites_filtred = pd.read_csv(sites_filtre, skiprows=1)
#        xsites = sites_filtred.Site_Name.values
#        xlat = []
#        xlon = []
#        for site in xsites:
#            xlat.append(sites['Latitude(decimal_degrees)'].values[(sites.Site_Name.values==site)])
#            xlon.append(sites['Longitude(decimal_degrees)'].values[(sites.Site_Name.values==site)])
#
#        xlat = np.array(xlat)
#        xlon = np.array(xlon)
#
#    else:
    xtimes = times
    xsites = sites.Site_Name.values
    xlat = sites['Latitude(decimal_degrees)'].values 
    xlon = sites['Longitude(decimal_degrees)'].values

#    xsites = sites.Site_Name[filtre_test].values
    date_start = dt.datetime(year_start, 1, 1)
    date_end = dt.datetime(year_end, 12, 31)
    ndates = (date_end - date_start).days + 1
#    xdates = np.array([[date_start+dt.timedelta(days=idate,hours=9), date_start+dt.timedelta(days=idate,hours=12), date_start+dt.timedelta(days=idate,hours=15)] for idate in range(ndates)]).ravel().astype('datetime64[ns]')
    xdates = np.array([[date_start+dt.timedelta(days=idate,hours=h) for h in times] for idate in range(ndates)]).ravel().astype('datetime64[ns]')

    merra_taua550 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_taua440 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_taua500 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_taua550 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_taua551 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_angstrom440 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_angstrom500 = np.zeros((len(xdates), len(xsites))) + np.NaN
#    merra_taua550 =         np.zeros((len(xtimes), len(xdates), len(xsites))) + np.NaN
#    aeronet_taua440 =       np.zeros((len(xtimes), len(xdates), len(xsites))) + np.NaN
#    aeronet_taua500 =       np.zeros((len(xtimes), len(xdates), len(xsites))) + np.NaN
#    aeronet_taua550 =       np.zeros((len(xtimes), len(xdates), len(xsites))) + np.NaN
#    aeronet_taua551 =       np.zeros((len(xtimes), len(xdates), len(xsites))) + np.NaN
#    aeronet_angstrom440 =   np.zeros((len(xtimes), len(xdates), len(xsites))) + np.NaN
#    aeronet_angstrom500 =   np.zeros((len(xtimes), len(xdates), len(xsites))) + np.NaN

    data_taua = xarray.Dataset({'aeronet_taua550':(('date','site'), aeronet_taua550), 'aeronet_taua551':(('date','site'), aeronet_taua551),'aeronet_taua440': (('date', 'site'), aeronet_taua440), 'aeronet_taua500': (('date', 'site'), aeronet_taua500), 'aeronet_angstrom440_675': (('date', 'site'), aeronet_angstrom440), 'aeronet_angstrom500_870': (('date', 'site'), aeronet_angstrom500), 'lat':(('site'), xlat), 'lon':(('site'), xlon), 'merra_taua550':(('date','site'), merra_taua550)}, coords={'date':xdates, 'site':xsites})
#    data_taua = xarray.Dataset({'aeronet_taua550':(('time','date','site'), aeronet_taua550), 'aeronet_taua551':(('time','date','site'), aeronet_taua551),'aeronet_taua440': (('time','date', 'site'), aeronet_taua440), 'aeronet_taua500': (('time','date', 'site'), aeronet_taua500), 'aeronet_angstrom440_675': (('time','date', 'site'), aeronet_angstrom440), 'aeronet_angstrom500_870': (('time','date', 'site'), aeronet_angstrom500), 'lat':(('site'), xlat), 'lon':(('site'), xlon), 'merra_taua550':(('time','date','site'), merra_taua550)}, coords={'time':xtimes, 'date':xdates, 'site':xsites})

    nsiteutil = 0
    t500 = 0
    t440 = 0
    list_sites = []
    for site_name in xsites:
#        if site_name != 'Cabauw':
#        if site_name == 'Gloria':
#        if site_name == 'ICIPE-Mbita':
#            continue
        print site_name
        aeronet_data, nrows = readAeronet(site_name, year_start, year_end, times)
        if nrows == 0:
            stat = "{} : rows = 0".format(site_name)
            continue
        else:
            nutil500 = sum((~np.isnan(aeronet_data['aot500'].values[0])) & (aeronet_data['aot500'].values != -999))
            nutil440 = sum((~np.isnan(aeronet_data['aot440'].values[0])) & (aeronet_data['aot440'].values != -999))
            stat = "{0} : rows = {1} : aod500 {2}/{4} ; aod440 {3}/{4}".format(site_name, nrows, nutil500, nutil440, aeronet_data['aot500'].values.shape[0])
            t500 += nutil500
            t440 += nutil440
            if nutil500 != 0 or nutil440 != 0:
                nsiteutil += 1
                list_sites.append(stat)
#        print stat


#        filtre_date = (aeronet_data.index-dt.datetime(year_start, 1, 1)).days
#        filtre_site = (xsites==site_name)

        data_taua['aeronet_taua550'].loc[aeronet_data.index,site_name] = aeronet_data.aot550 #np.reshape(aeronet_data['aot550'], (aeronet_data['aot550'].shape[0], 1))
        data_taua['aeronet_taua551'].loc[aeronet_data.index,site_name] = aeronet_data.aot551 #np.reshape(aeronet_data['aot551'], (aeronet_data['aot551'].shape[0], 1))
        data_taua['aeronet_taua440'].loc[aeronet_data.index,site_name] = aeronet_data.aot440 #np.reshape(aeronet_data['aot440'], (aeronet_data['aot440'].shape[0], 1))
        data_taua['aeronet_taua500'].loc[aeronet_data.index,site_name] = aeronet_data.aot500 #np.reshape(aeronet_data['aot500'], (aeronet_data['aot500'].shape[0], 1))
        data_taua['aeronet_angstrom440_675'].loc[aeronet_data.index,site_name] = aeronet_data['angstrom_440_675'] #np.reshape(aeronet_data['angstrom_440_675'], (aeronet_data['angstrom_440_675'].shape[0], 1))
        data_taua['aeronet_angstrom500_870'].loc[aeronet_data.index,site_name] = aeronet_data['angstrom_500_870'] #np.reshape(aeronet_data['angstrom_500_870'], (aeronet_data['angstrom_500_870'].shape[0], 1))
#        data_taua['aeronet_taua550'][filtre_date,filtre_site] = np.reshape(aeronet_data['aot550'], (aeronet_data['aot550'].shape[0], 1))
#        data_taua['aeronet_taua551'][filtre_date,filtre_site] = np.reshape(aeronet_data['aot551'], (aeronet_data['aot551'].shape[0], 1))
#        data_taua['aeronet_taua440'][filtre_date,filtre_site] = np.reshape(aeronet_data['aot440'], (aeronet_data['aot440'].shape[0], 1))
#        data_taua['aeronet_taua500'][filtre_date,filtre_site] = np.reshape(aeronet_data['aot500'], (aeronet_data['aot500'].shape[0], 1))
#        data_taua['aeronet_angstrom440_675'][filtre_date,filtre_site] = np.reshape(aeronet_data['angstrom_440_675'], (aeronet_data['angstrom_440_675'].shape[0], 1))
#        data_taua['aeronet_angstrom500_870'][filtre_date,filtre_site] = np.reshape(aeronet_data['angstrom_500_870'], (aeronet_data['angstrom_500_870'].shape[0], 1))

#    print "sites utiles : {} nd 500 : {} nd 440 :  {}".format(nsiteutil, t500, t440)
#    with open('sites_{}_{}.txt'.format(year_start, year_end),'w') as pf:
#        txt = '\n'.join(list_sites)
#        pf.write(txt)

    for date in data_taua.date:
        year = str(date.values)[:4]
        month = str(date.values)[5:7]
        day = str(date.values)[8:10]
        filename = '/rfs/data/MERRA2/aer_extinction/{0}/*{0}{1}{2}*.nc4'.format(year, month, day)
        filename = glob(filename)
        print filename
        if len(filename) != 1: 
            continue
        filename = filename[0]
        merra_date = date.values# + np.timedelta64(12, 'h')
        print merra_date


        merra_lut = readMerra(filename)

#        filtre_date = (data_taua.date == date)
        for site in data_taua.site:
#            filtre_site = (xsites==site.values)
#            aeronet = data_taua['aeronet_taua550'][filtre_date,filtre_site].values[0,0]
#            if np.isnan(aeronet):
#            if np.isnan(data_taua['aeronet_taua440'][filtre_date, filtre_site].values[0,0]) and np.isnan(data_taua['aeronet_taua500'][filtre_date, filtre_site].values[0,0]):
            if np.isnan(data_taua['aeronet_taua440'].loc[date, site].values) and np.isnan(data_taua['aeronet_taua500'].loc[date, site].values):
                continue
            lat = data_taua['lat'].loc[site].values
            lon = data_taua['lon'].loc[site].values

            merra_taua = merra_lut['TOTEXTTAU'][Idx(merra_date, fill_value='extrema'), Idx(lat, round=False, fill_value='extrema'), Idx(lon, round=False, fill_value='extrema')]
#            tmp = data_taua['merra_taua550'].values
#            tmp[filtre_date, filtre_site] = merra_taua
#            data_taua['merra_taua550'].values = tmp
            data_taua['merra_taua550'].loc[date, site] = merra_taua

#    fileout = '/rfs/proj/C3S/merra_aeronet_{}_{}_cabauw_v3.nc'.format(year_start, year_end)

    fileout = '/rfs/proj/C3S/validation_merra_aeronet/merra_aeronet_{}_hourly.nc'.format(year_start)
    data_taua.to_netcdf(fileout)

