#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest
import sys
sys.path.append('/mnt/qnaphome/bruno/Projets/smaccl')
from c3s_io_modis import load_modis
from ISmaccl import ISmaccl
import configparser
from c3s_lib import pre_merra2, pre_aer_models
from luts.luts import read_mlut

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
    merra_lut = pre_merra2(merra_aerosol, merra_ptwo)
    frac_aer_model = pre_aer_models(faer)
    dem = read_mlut(config['dem'])

    # test de l'interface smaccl
    sm = ISmaccl(config, platform='CPU')
    toc_data = sm.run(l2_data, merra_lut, dem, frac_aer_model, coeffs)

    print(toc_data)
    toc_data['SZA'] = l2_data['SZA']
    toc_data['VZA'] = l2_data['VZA']
    toc_data['SAA'] = l2_data['SAA']
    toc_data['VAA'] = l2_data['VAA']
    toc_data['lat'] = l2_data['lat']
    toc_data['lon'] = l2_data['lon']

    toc_data.to_netcdf(config['output'])

if __name__ == "__main__":
    test_run_smaccl()