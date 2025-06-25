#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import sys
import xarray as xa
from core import interpolate
sys.path.append('/mnt/qnaphome/bruno/Projets/smaccl/smaccl/')
from c3s_io_modis import load_modis
from ISmaccl import ISmaccl
import configparser
from c3s_lib import pre_merra2, pre_aer_models
#from luts.luts import read_mlut

#product = pytest.fixture(params=[
#    '/mnt/pve/cephfs/proj/CCI_VGT/MODIS/TERRA/A2019052.0950.061/MODIS_Terra_500_X19Y04_201902210950_333M.nc'
#])
def readConfig(configfile):
    cf = configparser.ConfigParser()
    cf.read(configfile)
    config = {}
    for k,v  in cf.items('Coefficients'):
       config[k] = float(v) 
    for k,v  in cf.items('Sizes'):
       config[k] = int(v) 
    for k,v in cf.items('Paths'):
        config[k] = v
    for k,v in cf.items('Sensor'):
        config[k] = v
    for k,v in cf.items('Output'):
        config[k] = eval(v)

    return config

def test_run_smaccl(): 
    product = '/mnt/pve/cephfs/proj/CCI_VGT/MODIS/TERRA/A2019052.0950.061/MODIS_Terra_500_X19Y04_201902210950_333M.nc'
    smaccoeffs = '/home/bruno/Projets/CCI_VGT/MODIS_TERRA_CODE/COEFFS/'
    merra_aerosol = "/archive2/data/MERRA2/aer_extinction/2019/MERRA2_400.tavg1_2d_aer_Nx.20190221.nc4"
    merra_ptwo = "/archive2/data/MERRA2/surf_pression_water_vapor/2019/MERRA2_400.tavg1_2d_slv_Nx.20190221.nc4"
    dem = '/archive2/data/DEM/GLOBE/GTOPO30_DZ_MLUT.nc'
    faer = "/home/bruno/Projets/CCI_VGT/MODIS_TERRA_CODE/ancillary/Aerosol_model_fraction.txt"
    bands = [1, 2, 3, 4, 5, 7]
    config = readConfig('modis_terra_config.cfg')
    l2_data, coeffs, sizes, tab_band_internal = load_modis(product, smaccoeffs, 0, -1, bands)
    l2_data, coeffs, sizes, tab_band_internal = load_modis(product, smaccoeffs, 0, sizes[0], bands) 
    l2_data['bands'] = tab_band_internal
    t0 = l2_data['mean-time-dec'].data
    frac_aer_model = pre_aer_models(faer)
    dem = xa.open_dataset(dem)
    elev = dem['elev']
    Delev = dem['Delev']
    elev_interp = interpolate.interp(elev, lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']))
    Delev_interp = interpolate.interp(Delev, lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']))
    dem = xa.Dataset({'elev': elev_interp, 'Delev': Delev_interp},
                       coords={'lat': l2_data['lat'], 'lon': l2_data['lon']})
    
    merra_aer = xa.open_dataset(merra_aerosol)
    merra_p2 = xa.open_dataset(merra_ptwo)

    tau  = interpolate.interp(merra_aer['TOTEXTTAU'], lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']), time=interpolate.Linear(l2_data['mean-time']))
    uh2o    = interpolate.interp(merra_p2['TQV'], lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']), time=interpolate.Linear(l2_data['mean-time']))
    uo3     = interpolate.interp(merra_p2['TO3'], lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']), time=interpolate.Linear(l2_data['mean-time']))
    p0      = interpolate.interp(merra_p2['SLP'], lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']), time=interpolate.Linear(l2_data['mean-time']))
    t10m    = interpolate.interp(merra_p2['T10M'], lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']), time=interpolate.Linear(l2_data['mean-time']))
    bc_frac = interpolate.interp(merra_aer['BCEXTTAU'] , lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']), time=interpolate.Linear(l2_data['mean-time']))
    du_frac = interpolate.interp(merra_aer['DUEXTTAU'] , lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']), time=interpolate.Linear(l2_data['mean-time']))
    oc_frac = interpolate.interp(merra_aer['OCEXTTAU'] , lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']), time=interpolate.Linear(l2_data['mean-time']))
    ss_frac = interpolate.interp(merra_aer['SSEXTTAU'] , lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']), time=interpolate.Linear(l2_data['mean-time']))
    su_frac = interpolate.interp(merra_aer['SUEXTTAU'] , lat=interpolate.Linear(l2_data['lat']), lon=interpolate.Linear(l2_data['lon']), time=interpolate.Linear(l2_data['mean-time']))

    merra = xa.Dataset({'TOTEXTTAU':tau, 'TQV':uh2o, 'TO3':uo3, 'SLP':p0, 'T10M':t10m, 'BC_FRAC':bc_frac/tau, 'DU_FRAC':du_frac/tau, 'SS_FRAC':ss_frac/tau, 'SU_FRAC':su_frac/tau, "OC_FRAC":oc_frac}, 
                       coords={'lat': l2_data['lat'], 'lon': l2_data['lon']})


    # test de l'interface smaccl
    sm = ISmaccl(config, platform='CPU')
    toc_data = sm.run(l2_data, merra, dem, frac_aer_model, coeffs)

#    print(toc_data)
    toc_data['SZA'] = l2_data['SZA']
    toc_data['VZA'] = l2_data['VZA']
    toc_data['SAA'] = l2_data['SAA']
    toc_data['VAA'] = l2_data['VAA']
    toc_data['lat'] = l2_data['lat']
    toc_data['lon'] = l2_data['lon']

    toc_data.to_netcdf(config['output'])

if __name__ == "__main__":
    test_run_smaccl()