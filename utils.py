# encoding: utf-8

import numpy as np
from glob import glob

type_coeff = [
#    ('bandname',    'U25'),
    ('ah2o',        'float32'),
    ('nh2o',        'float32'), 
    ('ao3',        'float32'), 
    ('no3',        'float32'), 
    ('ao2',        'float32'), 
    ('no2',        'float32'), 
    ('po2',        'float32'), 
    ('aco2',        'float32'), 
    ('nco2',        'float32'), 
    ('pco2',        'float32'), 
    ('ach4',        'float32'), 
    ('nch4',        'float32'), 
    ('pch4',        'float32'), 
    ('ano2',        'float32'), 
    ('nno2',        'float32'), 
    ('pno2',        'float32'), 
    ('aco',        'float32'), 
    ('nco',        'float32'), 
    ('pco',        'float32'), 
    ('a0s',        'float32'), 
    ('a1s',        'float32'), 
    ('a2s',        'float32'), 
    ('a3s',        'float32'), 
    ('a0T',        'float32'), 
    ('a1T',        'float32'), 
    ('a2T',        'float32'), 
    ('a3T',        'float32'), 
    ('taur',        'float32'), 
    ('a0taup',        'float32'), 
    ('a1taup',        'float32'), 
    ('wo',        'float32'), 
    ('gc',        'float32'), 
    ('a0P',        'float32'), 
    ('a1P',        'float32'), 
    ('a2P',        'float32'), 
    ('a3P',        'float32'), 
    ('a4P',        'float32'), 
    ('Resa1',        'float32'), 
    ('Resa2',        'float32'), 
    ('Resa3',        'float32'), 
    ('Resa4',        'float32'), 
    ('Resr1',        'float32'), 
    ('Resr2',        'float32'), 
    ('Resr3',        'float32'), 
    ('Rest1',        'float32'), 
    ('Rest2',        'float32'), 
    ('Rest3',        'float32'), 
    ('Rest4',        'float32')
  ]

def get_smac_coeffs(bands, bandidx):
    '''
    bands : a list of string containing band names to be processed, should be indentical to names in the COEFFS 
                        directory
    '''

    data = np.load(bands)
    if len(data.shape)==2:
        coeffs = np.zeros((len(bandidx), data.shape[1]), dtype=type_coeff, order='C')
        for i, idx in enumerate(bandidx):
            for i2, d in enumerate(data[idx-1]):
                coeffs[i,i2] = d.tolist()[1:]
    else:
        coeffs = np.zeros((len(bandidx)), dtype=type_coeff, order='C')
        for i, idx in enumerate(bandidx):
            for co in data:
                if int(co[0][-2:]) == idx:
                    coeffs[i] = co.tolist()[1:]
        
    return coeffs

def Ps(z,p0,T, g=9.801, R=287.058, lam=-0.006):
    T1 = np.log(R*T) - np.log(-R*lam*z+R*T)
    return p0*np.exp(-g/(R*lam)*T1) 

def dPsdz(z,p0,T, g=9.801, R=287.058, lam=-0.006):
    return g*Ps(z,p0,T, g=9.801, R=287.058, lam=-0.006)/(R*(T-lam*z))

if __name__=='__main__':
    filename = './COEFFS/s3b_olci_hitran2012_opac_cont_avg_v1.0.npy'
    filename = '/rfs/proj/C3S/SMAC_COEFFS/NOAA_07_smac_coeffs.npy'
#    filename = '/rfs/proj/C3S/SMAC_COEFFS/METOP_A_smac_coeffs.npy'
    filename = '/rfs/proj/C3S/SMAC_COEFFS/S3A_OLCI_smac_coeffs.npy'
    olci_idx = [2,3,4,5,6,7,8,9,10,11,12,16,17,18,21]
    olci_idx = [1]
    coeffs = get_smac_coeffs(filename, olci_idx)
#    print(coeffs[0,0])
#    print(len(coeffs.shape))
    filename = './COEFFS/s3b_olci_hitran2012_opac_cont_avg_v1.0.npy'
    coeffs_cont = get_smac_coeffs(filename, olci_idx)
    for i in range(len(coeffs[0,0])):
        print(coeffs[0,0][i], coeffs_cont[0][i])

#    files = glob('/rfs/proj/C3S/SMAC_COEFFS/*.npy')
#    for f in files:
#        print(f)
#        coeffs = get_smac_coeffs(filename, olci_idx)
#        print(coeffs[0,0])
