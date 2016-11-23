#!/usr/bin/env python
# encoding: utf-8

'''
SMAC-G
SMAC atmospheric correction using GPU
'''
from __future__ import print_function, division
import numpy as np
from os.path import dirname, realpath, join, exists
from pycuda.gpuarray import to_gpu, zeros as gpuzeros
import sys
if sys.version_info[:2] >= (3, 0):
        xrange = range


# set up directories
dir_root = dirname(realpath(__file__))
dir_src = join(dir_root, 'src/')
dir_bin = join(dir_root, 'bin/')
src_device = join(dir_src, 'device.cu')
binname =  join(dir_bin, 'sm.cubin')


type_coeff = [
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
    ('a0u',        'float32'), 
    ('a1u',        'float32'), 
    ('a2u',        'float32'), 
    ('a0s',        'float32'), 
    ('a1s',        'float32'), 
    ('a2s',        'float32'), 
    ('a3s',        'float32'), 
    ('a0T',        'float32'), 
    ('a1T',        'float32'), 
    ('a2T',        'float32'), 
    ('a3T',        'float32'), 
    ('taur',        'float32'), 
    ('sr',        'float32'), 
    ('a0taup',        'float32'), 
    ('a1taup',        'float32'), 
    ('wo',        'float32'), 
    ('gc',        'float32'), 
    ('a0P',        'float32'), 
    ('a1P',        'float32'), 
    ('a2P',        'float32'), 
    ('a3P',        'float32'), 
    ('a4P',        'float32'), 
    ('a5P',        'float32'), 
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

class coeff:
  def __init__(self,smac_filename):
    with file(smac_filename) as f:
      lines=f.readlines()
    #H20
    temp=lines[0].strip().split()
    self.ah2o=float(temp[0])
    self.nh2o=float(temp[1])
    #O3
    temp=lines[1].strip().split()
    self.ao3=float(temp[0])
    self.no3=float(temp[1])
    #O2
    temp=lines[2].strip().split()
    self.ao2=float(temp[0])
    self.no2=float(temp[1])
    self.po2=float(temp[2])
    #CO2
    temp=lines[3].strip().split()
    self.aco2=float(temp[0])
    self.nco2=float(temp[1])
    self.pco2=float(temp[2])
    #NH4
    temp=lines[4].strip().split()
    self.ach4=float(temp[0])
    self.nch4=float(temp[1])
    self.pch4=float(temp[2])
    #NO2
    temp=lines[5].strip().split()
    self.ano2=float(temp[0])
    self.nno2=float(temp[1])
    self.pno2=float(temp[2])
    #NO2
    temp=lines[6].strip().split()
    self.aco=float(temp[0])
    self.nco=float(temp[1])
    self.pco=float(temp[2])

    #rayleigh and aerosol scattering
    temp=lines[7].strip().split()
    self.a0s=float(temp[0])
    self.a1s=float(temp[1])
    self.a2s=float(temp[2])
    self.a3s=float(temp[3])
    temp=lines[8].strip().split()
    self.a0T=float(temp[0])
    self.a1T=float(temp[1])
    self.a2T=float(temp[2])
    self.a3T=float(temp[3])
    temp=lines[9].strip().split()
    self.taur=float(temp[0])
    self.sr=float(temp[0])
    temp=lines[10].strip().split()
    self.a0taup  = float(temp[0])
    self.a1taup  = float(temp[1])
    temp=lines[11].strip().split()
    self.wo      = float(temp[0])
    self.gc      = float(temp[1])
    temp=lines[12].strip().split()
    self.a0P     = float(temp[0])
    self.a1P     = float(temp[1])
    self.a2P     = float(temp[2])
    temp=lines[13].strip().split()
    self.a3P     = float(temp[0])
    self.a4P     = float(temp[1])
    temp=lines[14].strip().split()
    self.Rest1   = float(temp[0])
    self.Rest2   = float(temp[1])
    temp=lines[15].strip().split()
    self.Rest3   = float(temp[0])
    self.Rest4   = float(temp[1])
    temp=lines[16].strip().split()
    self.Resr1   = float(temp[0])
    self.Resr2   = float(temp[1])
    self.Resr3   = float(temp[2])
    temp=lines[17].strip().split()
    self.Resa1   = float(temp[0])
    self.Resa2   = float(temp[1])
    temp=lines[18].strip().split()
    self.Resa3   = float(temp[0])
    self.Resa4   = float(temp[1])


class Smacg(object):

    def __init__(self):

        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        from pycuda.driver import module_from_buffer

        if exists(src_device):

            # load device.cu
            src_device_content = open(src_device).read()

            # kernel compilation
            self.mod = SourceModule(src_device_content,
                               nvcc='nvcc',
                               no_extern_c=True,
                               cache_dir='/tmp/',
                               include_dirs=[dir_src,
                                            join(dir_src, 'incRNGs/Random123/')])

        elif exists(binname):
            # load existing binary
            print('Loading binary', binname)
            self.mod = module_from_buffer(open(binname, 'rb').read())

        else:
            raise IOError('Could not find {} or {}.'.format(src_device, binname))

        # load the kernel
        self.kernel = self.mod.get_function('smacg')


    def copy_to_device(self, name, scalar, dtype):
        from pycuda.driver import memcpy_htod
        memcpy_htod(self.mod.get_global(name)[0], np.array([scalar], dtype=dtype))


    def run(self, bands, tetas, tetav, phis, phiv,
                uh2o, uo3, taup550, pression, rtoa,
                XBLOCK=128, XGRID=128, NBLOOP=1):

        '''
        Run an atmospheric correction run using SMAC

        Arguments:

            - bands : a list of string containing band names to be processed, should be indentical to names in the COEFFS 
                    directory

            - tetas: SZA float32 arrays of dimension (XBLOCK,XGRID,Z) where Z is 3rd dimension of pixels

            - tetav: VZA float32 arrays of dimension (XBLOCK,XGRID,Z) where Z is 3rd dimension of pixels

            - phis : SAA float32 arrays of dimension (XBLOCK,XGRID,Z) where Z is 3rd dimension of pixels

            - phiv : VAA float32 arrays of dimension (XBLOCK,XGRID,Z) where Z is 3rd dimension of pixels

            - uh2o : Water vapour column float32 arrays of dimension (XBLOCK,XGRID,Z) where Z is 3rd dimension of pixels

            - uo3  : Ozone column float32 arrays of dimension (XBLOCK,XGRID,Z) where Z is 3rd dimension of pixels

            - taup550  : AOT at 550 nm float32 arrays of dimension (XBLOCK,XGRID,Z) where Z is 3rd dimension of pixels

            - pression : Surface pressure float32 arrays of dimension (XBLOCK,XGRID,Z) where Z is 3rd dimension of pixels

            - rtoa : TOA reflectance float32 arrays of dimension (XBLOCK,XGRID,Z, NB) where Z is 3rd dimension of pixels,
                        and NB is the number of bands

            - XBLOCK and XGRID: control the number of blocks and grid size for
              the GPU execution

            - NBLOOP: number of runs within a thread for the same pixel (should be used for Monte Carlo draws)

        '''

        NBAND = len(bands)
        self.coeffs = np.zeros((NBAND), dtype=type_coeff)

        for ib,band in enumerate(bands):
            co = coeff(band)
            for k in co.__dict__.keys():
                self.coeffs[k] = co.__dict__[k]

        shp = rtoa.shape
        assert shp[-1] == NBAND
        if (rtoa.ndim == 4) :
            NZ  = shp[-2]
        else : NZ=1

        self.copy_to_device('NBLOOPd', NBLOOP, np.uint32)
        self.copy_to_device('XBLOCKd', XBLOCK, np.uint32)
        self.copy_to_device('XGRIDd' , XGRID,  np.uint32)
        self.copy_to_device('NZd'    , NZ,     np.uint32)
        self.copy_to_device('NBANDd' , NBAND,  np.uint32)

        #output arrays
        rsurf    = gpuzeros(shp, dtype=np.float32)
        Jrtoa    = gpuzeros(shp, dtype=np.float32)
        Juo3     = gpuzeros(shp, dtype=np.float32)
        Juh2o    = gpuzeros(shp, dtype=np.float32)
        Jpre     = gpuzeros(shp, dtype=np.float32)
        Jtaup    = gpuzeros(shp, dtype=np.float32)


        # run
        self.kernel(to_gpu(self.coeffs), 
        to_gpu(tetas) , 
        to_gpu(tetav) , 
        to_gpu(phis)  , 
        to_gpu(phiv)  , 
        to_gpu(uh2o.astype(np.float32)) , 
        to_gpu(uo3.astype(np.float32)) , 
        to_gpu(taup550.astype(np.float32)) , 
        to_gpu(pression.astype(np.float32)), 
        to_gpu(rtoa), 
        rsurf,
        Jrtoa,
        Juo3,
        Juh2o,
        Jpre,
        Jtaup,
        block=(XBLOCK, 1, 1), grid=(XGRID, 1, 1)
        )

        return ( rsurf.get(), Jrtoa.get(), Juo3.get(), Juh2o.get(), Jpre.get(), Jtaup.get() )
