from glob import glob
from tools.luts import MLUT, Idx#, read_mlut
import xarray
import numpy as np
from read_file import index_to_datetime
import pandas as pd
import datetime as dt
from os.path import exists

def comp_aot550(data, year, conv):
    filtre_date = ((data.index >= dt.datetime(year, 1, 1)) & (data.index <= dt.datetime(year, 12, 31)))
    ndata = np.sum(filtre_date)
    print ndata

    aot550 =            np.zeros((ndata)) + np.NaN

    a440_675 = data[conv['ANG440-675']][filtre_date]
#    aot550_440 = ((550./440.)**(-data['440-675_Angstrom_Exponent']))*data.AOD_440nm
#    filtre = ((~np.isnan(aot550_440)) & (~np.isnan(data['440-675_Angstrom_Exponent'])))
    aot550_440 = ((550./440.)**(-a440_675))*data[conv['AOT440']][filtre_date]
    filtre = ((~np.isnan(aot550_440)) & (~np.isnan(a440_675)))
    aot550[filtre] = aot550_440[filtre]

    a500_870 = data[conv['ANG500-870']][filtre_date]
#    aot550_500 = ((550./500.)**(-data['500-870_Angstrom_Exponent']))*data.AOD_500nm
#    filtre = ((~np.isnan(aot550_500)) & (~np.isnan(data['500-870_Angstrom_Exponent'])))
    aot550_500 = ((550./500.)**(-a500_870))*data[conv['AOT500']][filtre_date]
    filtre = ((~np.isnan(aot550_500)) & (~np.isnan(a500_870)))
    aot550[filtre] = aot550_500[filtre]

#    filtre = (~np.isnan(data.AOD_551nm))
#    aot550[filtre] = data.AOD_551nm[filtre]
    filtre = (~np.isnan(data[conv['AOT551']][filtre_date]))
    aot550[filtre] = data[conv['AOT551']][filtre_date][filtre]

    return aot550

def meanAeronet(data, hour, delta, conv):
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

#        a440 = data.AOD_440nm[date_min:date_max]
        try:
            a440 = data[conv['AOT440']][date_min:date_max]
            a440[(a440==-999)] = np.NaN
            aot440.append(np.nanmean(a440))
        except:
            aot440.append(np.NaN)

        #a500 = data.AOD_500nm[date_min:date_max]
        try:
            a500 = data[conv['AOT500']][date_min:date_max]
            a500[(a500==-999)] = np.NaN
            aot500.append(np.nanmean(a500))
        except:
            aot500.append(np.NaN)

        #a551 = data.AOD_551nm[date_min:date_max]
        try:
            a551 = data[conv['AOT551']][date_min:date_max]
            a551[(a551==-999)] = np.NaN
            aot551.append(np.mean(a551))
        except:
            aot551.append(np.NaN)

 #       a440_675 = data['440-675_Angstrom_Exponent'][date_min:date_max]
        try:
            a440_675 = data[conv['ANG440-675']][date_min:date_max]
            a440_675[(a440_675==-999)] = np.NaN
            angstrom440_675.append(np.mean(a440_675))
        except:
            angstrom440_675.append(np.NaN)

#        a500_870 = data['500-870_Angstrom_Exponent'][date_min:date_max]
        try:
            a500_870 = data[conv['ANG500-870']][date_min:date_max]
            a500_870[(a500_870==-999)] = np.NaN
            angstrom500_870.append(np.mean(a500_870))
        except:
            angstrom500_870.append(np.NaN)

        dates.append(dt.datetime(date.year, date.month, date.day, hour, 0, 0))

    date_pandas = pd.DatetimeIndex(dates)
#    data_pandas = pd.DataFrame(data={'AOD_440nm':np.array(aot440), 'AOD_500nm':np.array(aot500), 'AOD_551nm':np.array(aot551), '440-675_Angstrom_Exponent':np.array(angstrom440_675), '500-870_Angstrom_Exponent':angstrom500_870}, index=date_pandas)
    data_pandas = pd.DataFrame(data={conv['AOT440']:np.array(aot440), conv['AOT500']:np.array(aot500), conv['AOT551']:np.array(aot551), conv['ANG440-675']:np.array(angstrom440_675), conv['ANG500-870']:angstrom500_870}, index=date_pandas)

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

def readAeronet(site_name, year, times):
    AODversion = '15'
    AODversion = '20'
    AVG = '20' #daily average
    AVG = '10' #all points

    year1 = str(year)
    month1 = '01'
    day1 = '01'
    year2 = str(year)
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
    if AODversion=='15':
        csv_url = 'https://aeronet.gsfc.nasa.gov/cgi-bin/print_web_data_v3?site=' + site_name + '&year=' + year1 + '&month=' + month1 + '&day=' + day1 + '&year2=' + year2 + '&month2=' + month2 + '&day2=' + day2 + '&AOD' + AODversion + '=1&AVG=' + AVG + '&if_no_html=1'
    elif AODversion=='20':
        csv_url = '/rfs/proj/C3S/aeronet/AOT/LEV20/ALL_POINTS/920801_171111_{}.lev20'.format(site_name)

    print csv_url
#    csv_data = pd.read_csv(csv_url, error_bad_lines=False, header=4) #, names=names_col, usecols=np.arange(len(names_col)))
    try:
        if AVG=='20':
#            times = [12]
            csv_data = pd.read_csv(csv_url, error_bad_lines=False, header=5) #, names=names_col, usecols=np.arange(len(names_col)))
        elif AODversion=='20':
            conv = {'AOT440':'AOT_440', 'AOT500':'AOT_500', 'AOT551':'AOT_551', 'ANG440-675':'440-675Angstrom','ANG500-870':'500-870Angstrom'}
            if exists(csv_url):
                csv_data = pd.read_csv(csv_url, error_bad_lines=False, header=4) #, names=names_col, usecols=np.arange(len(names_col)))
                csv_data = index_to_datetime(csv_data)
                filtre_date = ((csv_data.index >= dt.datetime(year, 1, 1)) & (csv_data.index <= dt.datetime(year, 12, 31)))
                if np.sum(filtre_date)!=0:
#                    csv_data =  csv_data.loc[dt.datetime(year, 1, 1):dt.datetime(year, 12, 31)]
                    csv_data =  csv_data.loc[filtre_date]
                    aeronet_data = []
                    for time in times:
                        aeronet_data.append( meanAeronet(csv_data, time, 30, conv))
                    csv_data = pd.concat(aeronet_data)
                else:
                    dateindex = pd.date_range('1/1/{}'.format(year), '31/12/{}'.format(year), freq='H')
                    tmp = np.zeros(dateindex.shape) + np.NaN
                    csv_data = pd.DataFrame({'AOT_500':tmp, 'AOT_440':tmp, 'AOT_550':tmp, 'AOT_551':tmp, '440-675Angstrom':tmp, '500-870Angstrom':tmp}, index=dateindex)
            else:
                dateindex = pd.date_range('1/1/{}'.format(year), '31/12/{}'.format(year), freq='H')
                tmp = np.zeros(dateindex.shape) + np.NaN
                csv_data = pd.DataFrame({'AOT_500':tmp, 'AOT_440':tmp, 'AOT_550':tmp, 'AOT_551':tmp, '440-675Angstrom':tmp, '500-870Angstrom':tmp}, index=dateindex)

#            filtre_date = ((csv_data.index >= dt.datetime(year, 1, 1)) & (csv_data.index <= dt.datetime(year, 12, 31)))
#            if np.sum(filtre_date)==0:
#                dateindex = pd.date_range('1/1/{}'.format(year), '31/12/{}'.format(year), freq='H')
#                tmp = np.zeros(dateindex.shape) + np.NaN
#                csv_data = pd.DataFrame({'AOT_500':tmp, 'AOT_440':tmp, 'AOT_550':tmp, 'AOT_551':tmp, '440-675Angstrom':tmp, '500-870Angstrom':tmp}, index=dateindex)
#            else:
#                aeronet_data = []
#                for time in times:
#                    aeronet_data.append( meanAeronet(csv_data, time, 30, conv))
        else:
            csv_data = pd.read_csv(csv_url, error_bad_lines=False, header=None, skiprows=6, names=names_col, usecols=np.arange(len(names_col)))
            csv_data = index_to_datetime(csv_data)
            aeronet_data = []
            conv = {'AOT440':'AOD_440nm', 'AOT500':'AOD_500nm', 'AOT551':'AOD_551nm', 'ANG440-675':'440-675_Angstrom_Exponent','ANG500-870':'500-870_Angstrom_Exponent'}
            for time in times:
                aeronet_data.append( meanAeronet(csv_data, time, 30, conv))
#            aeronet_data_9 = meanAeronet(csv_data, 9, 30)
#            aeronet_data_12 = meanAeronet(csv_data, 12, 30)
#            aeronet_data_15 = meanAeronet(csv_data, 15, 30)
#            frames = [aeronet_data_9, aeronet_data_12, aeronet_data_15]
#            csv_data = pd.concat(frames)
            csv_data = pd.concat(aeronet_data)

    except pd.errors.ParserError:
        return None, 0


    filtre_date = ((csv_data.index >= dt.datetime(year, 1, 1)) & (csv_data.index <= dt.datetime(year, 12, 31)))
    if AODversion=='20':
        print csv_data
        aot440 = csv_data.AOT_440.values[filtre_date]
        aot500 = csv_data.AOT_500.values[filtre_date]
        aot551 = csv_data.AOT_551.values[filtre_date]
        angstrom440_675 = csv_data['440-675Angstrom'].values[filtre_date]
        angstrom500_870 = csv_data['500-870Angstrom'].values[filtre_date]
    else:
        aot440 = csv_data.AOD_440nm.values[filtre_date]
        aot500 = csv_data.AOD_500nm.values[filtre_date]
        aot551 = csv_data.AOD_551nm.values[filtre_date]
        angstrom440_675 = csv_data['440-675_Angstrom_Exponent'].values[filtre_date]
        angstrom500_870 = csv_data['500-870_Angstrom_Exponent'].values[filtre_date]
    aot550 = comp_aot550(csv_data, year, conv)
    data = {'aot550': aot550, 'aot551': aot551, 'aot500':aot500, 'aot440':aot440, 'angstrom_440_675':angstrom440_675, 'angstrom_500_870':angstrom500_870}

    return pd.DataFrame(data=data, index=csv_data.index[filtre_date]), csv_data.shape[0]

if __name__=="__main__":
    year = 1998

#    times = [9, 12, 15]
    times = range(24)


    sites_file='/rfs/proj/C3S/aeronet_locations.txt'
    sites = pd.read_csv(sites_file, skiprows=1)

    xtimes = times
    xsites = sites.Site_Name.values
    xlat = sites['Latitude(decimal_degrees)'].values 
    xlon = sites['Longitude(decimal_degrees)'].values

    date_start = dt.datetime(year, 1, 1)
    date_end = dt.datetime(year, 12, 31)
    ndates = (date_end - date_start).days + 1
    xdates = np.array([[date_start+dt.timedelta(days=idate,hours=h) for h in times] for idate in range(ndates)]).ravel().astype('datetime64[ns]')

    merra_taua550 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_taua440 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_taua500 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_taua550 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_taua551 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_angstrom440 = np.zeros((len(xdates), len(xsites))) + np.NaN
    aeronet_angstrom500 = np.zeros((len(xdates), len(xsites))) + np.NaN

    data_taua = xarray.Dataset({'aeronet_taua550':(('date','site'), aeronet_taua550), 'aeronet_taua551':(('date','site'), aeronet_taua551),'aeronet_taua440': (('date', 'site'), aeronet_taua440), 'aeronet_taua500': (('date', 'site'), aeronet_taua500), 'aeronet_angstrom440_675': (('date', 'site'), aeronet_angstrom440), 'aeronet_angstrom500_870': (('date', 'site'), aeronet_angstrom500), 'lat':(('site'), xlat), 'lon':(('site'), xlon), 'merra_taua550':(('date','site'), merra_taua550)}, coords={'date':xdates, 'site':xsites})

    nsiteutil = 0
    t500 = 0
    t440 = 0
    list_sites = []
    for site_name in xsites:
#        if site_name != 'Cabauw':
#        if site_name == 'Gloria':
#        if site_name == 'ICIPE-Mbita':
#        if site_name != 'GSFC':
#            continue
        print site_name
        aeronet_data, nrows = readAeronet(site_name, year, times)

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


#        filtre_date = (aeronet_data.index-dt.datetime(year_start, 1, 1)).dayscoef_VGT2_B0_DES.dat
#        filtre_site = (xsites==site_name)

        data_taua['aeronet_taua550'].loc[aeronet_data.index,site_name] = aeronet_data.aot550 #np.reshape(aeronet_data['aot550'], (aeronet_data['aot550'].shape[0], 1))
        data_taua['aeronet_taua551'].loc[aeronet_data.index,site_name] = aeronet_data.aot551 #np.reshape(aeronet_data['aot551'], (aeronet_data['aot551'].shape[0], 1))
        data_taua['aeronet_taua440'].loc[aeronet_data.index,site_name] = aeronet_data.aot440 #np.reshape(aeronet_data['aot440'], (aeronet_data['aot440'].shape[0], 1))
        data_taua['aeronet_taua500'].loc[aeronet_data.index,site_name] = aeronet_data.aot500 #np.reshape(aeronet_data['aot500'], (aeronet_data['aot500'].shape[0], 1))
        data_taua['aeronet_angstrom440_675'].loc[aeronet_data.index,site_name] = aeronet_data['angstrom_440_675'] #np.reshape(aeronet_data['angstrom_440_675'], (aeronet_data['angstrom_440_675'].shape[0], 1))
        data_taua['aeronet_angstrom500_870'].loc[aeronet_data.index,site_name] = aeronet_data['angstrom_500_870'] #np.reshape(aeronet_data['angstrom_500_870'], (aeronet_data['angstrom_500_870'].shape[0], 1))

#    v = not np.isnan(data_taua.aeronet_taua550.loc[:,'GSFC'].values)
    date = date_start
    # loop year
    while date < date_end:

#        print date
        year = date.year
        month = date.month
        day = date.day
#        print year, month, day 
        filename = '/rfs/data/MERRA2/aer_extinction/{0}/*{0}{1:02}{2:02}*.nc4'.format(year, month, day)
        filename = glob(filename)
#        print filename
        if len(filename) != 1: 
            continue
        
        filename = filename[0]
        print filename
        merra_lut = readMerra(filename)

        hour_start = np.datetime64(dt.datetime(year,month,day, 0, 0))
        hour_end = np.datetime64(dt.datetime(year,month,day,23,59))
        filtre = ((data_taua.date>=hour_start) & (data_taua.date<=hour_end))
        date_aeronet = data_taua.date[filtre]
        # loop hour
        for datetime in date_aeronet:
            tmp =  str(datetime.values).split('T')[1].split(':')
            merra_hour = float(tmp[0]) + float(tmp[1])/60.

#            merra_date = datetime.values
#            print datetime, merra_date
        # loop site
            for site in data_taua.site:
#                if site != 'GSFC':
#                    continue

#                if np.isnan(data_taua['aeronet_taua440'].loc[date, site].values) and np.isnan(data_taua['aeronet_taua500'].loc[date, site].values):
                if np.isnan(data_taua['aeronet_taua550'].loc[datetime, site].values):
                    continue
                lat = data_taua['lat'].loc[site].values
                lon = data_taua['lon'].loc[site].values
                merra_taua = merra_lut['TOTEXTTAU'][Idx(merra_hour, fill_value='extrema'), Idx(lat, round=False, fill_value='extrema'), Idx(lon, round=False, fill_value='extrema')]
                data_taua['merra_taua550'].loc[datetime, site] = merra_taua

        date += dt.timedelta(days=1)
#    filtre_date = ((data_taua.date >= np.datetime64('1998-01-01')) & (data_taua.date <= np.datetime64('1998-01-02')))
    fileout = '/rfs/proj/C3S/validation_merra_aeronet/merra_aeronet_{}_hourly_aeronet2.0.nc'.format(year)
    data_taua.to_netcdf(fileout)
#    exit(0)
#    for date in data_taua.date:
#        year = str(date.values)[:4]
#        month = str(date.values)[5:7]
#        day = str(date.values)[8:10]
#        filename = '/rfs/data/MERRA2/aer_extinction/{0}/*{0}{1}{2}*.nc4'.format(year, month, day)
#        filename = glob(filename)
#        print filename
#        if len(filename) != 1: 
#            continue
#        filename = filename[0]
#        merra_date = date.values# + np.timedelta64(12, 'h')
#        print merra_date
#        continue
#
#
#        merra_lut = readMerra(filename)
#
##        filtre_date = (data_taua.date == date)
#        for site in data_taua.site:
##            filtre_site = (xsites==site.values)
##            aeronet = data_taua['aeronet_taua550'][filtre_date,filtre_site].values[0,0]
##            if np.isnan(aeronet):
##            if np.isnan(data_taua['aeronet_taua440'][filtre_date, filtre_site].values[0,0]) and np.isnan(data_taua['aeronet_taua500'][filtre_date, filtre_site].values[0,0]):
#            if np.isnan(data_taua['aeronet_taua440'].loc[date, site].values) and np.isnan(data_taua['aeronet_taua500'].loc[date, site].values):
#                continue
#            lat = data_taua['lat'].loc[site].values
#            lon = data_taua['lon'].loc[site].values
#
#            merra_taua = merra_lut['TOTEXTTAU'][Idx(merra_date, fill_value='extrema'), Idx(lat, round=False, fill_value='extrema'), Idx(lon, round=False, fill_value='extrema')]
##            tmp = data_taua['merra_taua550'].values
##            tmp[filtre_date, filtre_site] = merra_taua
##            data_taua['merra_taua550'].values = tmp
#            data_taua['merra_taua550'].loc[date, site] = merra_taua
#
##    fileout = '/rfs/proj/C3S/merra_aeronet_{}_{}_cabauw_v3.nc'.format(year_start, year_end)
#
#    fileout = '/rfs/proj/C3S/validation_merra_aeronet/merra_aeronet_{}_hourly.nc'.format(year_start)
#    data_taua.to_netcdf(fileout)
#
