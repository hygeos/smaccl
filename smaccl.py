# encoding: utf-8
################################################################
# 
# SMAC-CL
# SMAC atmospheric correction using CPU
#
################################################################
from __future__ import print_function, division
import numpy as np
import os
from os.path import dirname, realpath, join, exists
import pyopencl as cl
import sys
if sys.version_info[:2] >= (3, 0):
        xrange = range

__module__    = "smaccl.py"    
__version__   = "1.01.00"

print(__module__ + ' ' + __version__)

# set up directories
dir_root = dirname(realpath(__file__))
dir_src = join(dir_root, 'src/')
dir_bin = join(dir_root, 'bin/')
src_device = join(dir_src, 'devicecl.cl')
binname =  join(dir_bin, 'smac.clbin')
#dir_tmp = join(dir_root, 'tmp/')
dir_tmp = '/tmp/'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

type_coeff = [
    ('bandname',   'U25'),
    ('ah2o',       'float32'),
    ('nh2o',       'float32'), 
    ('ao3',        'float32'), 
    ('no3',        'float32'), 
    ('ao2',        'float32'), 
    ('no2',        'float32'), 
    ('po2',        'float32'), 
    ('aco2',       'float32'), 
    ('nco2',       'float32'), 
    ('pco2',       'float32'), 
    ('ach4',       'float32'), 
    ('nch4',       'float32'), 
    ('pch4',       'float32'), 
    ('ano2',       'float32'), 
    ('nno2',       'float32'), 
    ('pno2',       'float32'), 
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
    ('taur',       'float32'), 
    ('a0taup',     'float32'), 
    ('a1taup',     'float32'), 
    ('wo',         'float32'), 
    ('gc',         'float32'), 
    ('a0P',        'float32'), 
    ('a1P',        'float32'), 
    ('a2P',        'float32'), 
    ('a3P',        'float32'), 
    ('a4P',        'float32'), 
    ('Resa1',      'float32'), 
    ('Resa2',      'float32'), 
    ('Resa3',      'float32'), 
    ('Resa4',      'float32'), 
    ('Resr1',      'float32'), 
    ('Resr2',      'float32'), 
    ('Resr3',      'float32'), 
    ('Rest1',      'float32'), 
    ('Rest2',      'float32'), 
    ('Rest3',      'float32'), 
    ('Rest4',      'float32'),
    ('f1d0',      'float32'),
    ('f1d1',      'float32'),
    ('f1d2',      'float32'),
    ('f2d0',      'float32'),
    ('f2d1',      'float32'),
    ('f2d2',      'float32'),
    ('f1b0',      'float32'),
    ('f1b1',      'float32'),
    ('f1b2',      'float32'),
    ('f2b0',      'float32'),
    ('f2b1',      'float32'),
    ('f2b2',      'float32')
  ]


type_coeff_reduced = [
#    ('bandname',  'U25'),
    ('ah2o',       'float32'),
    ('nh2o',       'float32'), 
    ('ao3',        'float32'), 
    ('no3',        'float32'), 
    ('ao2',        'float32'), 
    ('no2',        'float32'), 
    ('po2',        'float32'), 
    ('aco2',       'float32'), 
    ('nco2',       'float32'), 
    ('pco2',       'float32'), 
    ('ach4',       'float32'), 
    ('nch4',       'float32'), 
    ('pch4',       'float32'), 
    ('ano2',       'float32'), 
    ('nno2',       'float32'), 
    ('pno2',       'float32'), 
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
    ('taur',       'float32'), 
    ('a0taup',     'float32'), 
    ('a1taup',     'float32'), 
    ('wo',         'float32'), 
    ('gc',         'float32'), 
    ('a0P',        'float32'), 
    ('a1P',        'float32'), 
    ('a2P',        'float32'), 
    ('a3P',        'float32'), 
    ('a4P',        'float32'), 
    ('Resa1',      'float32'), 
    ('Resa2',      'float32'), 
    ('Resa3',      'float32'), 
    ('Resa4',      'float32'), 
    ('Resr1',      'float32'), 
    ('Resr2',      'float32'), 
    ('Resr3',      'float32'), 
    ('Rest1',      'float32'), 
    ('Rest2',      'float32'), 
    ('Rest3',      'float32'), 
    ('Rest4',      'float32'),
    ('f1d0',      'float32'),
    ('f1d1',      'float32'),
    ('f1d2',      'float32'),
    ('f2d0',      'float32'),
    ('f2d1',      'float32'),
    ('f2d2',      'float32'),
    ('f1b0',      'float32'),
    ('f1b1',      'float32'),
    ('f1b2',      'float32'),
    ('f2b0',      'float32'),
    ('f2b1',      'float32'),
    ('f2b2',      'float32')
  ]

type_coeff_old = [
    ('ah2o',       'float32'),
    ('nh2o',       'float32'), 
    ('ao3',        'float32'), 
    ('no3',        'float32'), 
    ('ao2',        'float32'), 
    ('no2',        'float32'), 
    ('po2',        'float32'), 
    ('aco2',       'float32'), 
    ('nco2',       'float32'), 
    ('pco2',       'float32'), 
    ('ach4',       'float32'), 
    ('nch4',       'float32'), 
    ('pch4',       'float32'), 
    ('ano2',       'float32'), 
    ('nno2',       'float32'), 
    ('pno2',       'float32'), 
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
    ('taur',       'float32'), 
    ('sr',         'float32'), 
    ('a0taup',     'float32'), 
    ('a1taup',     'float32'), 
    ('wo',         'float32'), 
    ('gc',         'float32'), 
    ('a0P',        'float32'), 
    ('a1P',        'float32'), 
    ('a2P',        'float32'), 
    ('a3P',        'float32'), 
    ('a4P',        'float32'), 
    ('Resa1',      'float32'), 
    ('Resa2',      'float32'), 
    ('Resa3',      'float32'), 
    ('Resa4',      'float32'), 
    ('Resr1',      'float32'), 
    ('Resr2',      'float32'), 
    ('Resr3',      'float32'), 
    ('Rest1',      'float32'), 
    ('Rest2',      'float32'), 
    ('Rest3',      'float32'), 
    ('Rest4',      'float32')
  ]


def get_smac_coeffs(coeff_array_file, bandidx=None):
    '''
    coeff_array_file : a filename of a saved numpy array (.npy) containing all coefficiens,
    bandidx :  list of index of bands to be extracted
    '''

    data = np.load(coeff_array_file)
    if bandidx is None:
        bandidx = np.arange(data.shape[0])

    if len(data.shape)==2:
        coeffs = np.zeros((len(bandidx), data.shape[1]), dtype=type_coeff_reduced, order='C')
        for i, idx in enumerate(bandidx):
            for i2, d in enumerate(data[idx]):
                coeffs[i,i2] = d.tolist()[1:]
    else:
        coeffs = np.zeros((len(bandidx)), dtype=type_coeff_reduced, order='C')
        for i, idx in enumerate(bandidx):
            for co in data:
                if int(co[0][-2:]) == idx:
                    coeffs[i] = co.tolist()[1:]
        
    return coeffs


class coeff:
  def __init__(self,smac_filename):
    with open(smac_filename) as f:
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


def get_smac_coeffs_fromtxt(bands):
    '''
    OLD definition of SMAC COEFS computed by third party
    just for back compatibility
    bands : a list of string containing band names to be processed, 
    should be indentical to names in the COEFFS directory
    '''
    NBAND = len(bands)
    coeffs = np.zeros((NBAND), dtype=type_coeff_old, order='C')

    for ib,band in enumerate(bands):
        co = coeff(band)
        for k in co.__dict__.keys():
            coeffs[k][ib] = co.__dict__[k] 

    return coeffs


class Smaccl(object):

    def __init__(self, platform='GPU'):

#        print("....  testsmaccl1 class __init__ ")

        if exists(src_device):

            #            print("....  testsmaccl1 class __init__ exists source")
            
#            print ('... Obtain an OpenCL platform')
#            ####Set the environment variable PYOPENCL_CTX='0' to avoid being asked again.
#            ## Step #1. Obtain an OpenCL platform.
#            self.clplatform = cl.get_platforms()[1]
#            
#            
#            ## It would be necessary to add some code to check the check the support for
#            ## the necessary platform extensions with platform.extensions
#            
#            print ('... Obtain a device id')
#            ## Step #2. Obtain a device id for at least one device (accelerator).
#            self.cldevice = self.clplatform.get_devices()[0]
#            
#            print(self.cldevice)
#            ## It would be necessary to add some code to check the check the support for
#            ## the necessary device extensions with device.extensions
#            print ('... Create a context for the selected device')
#            ## Step #3. Create a context for the selected device.
#            self.clcontext = cl.Context([self.cldevice])
#            print(self.clcontext)
            # set up directories
#            print ('... Get a queue')
            #self.clqueue = cl.CommandQueue(self.clcontext)
#            self.clqueue = cl.CommandQueue(self.clcontext, properties=cl.command_queue_properties.PROFILING_ENABLE)
            status = self.set_queue(platform)
            if not(status):
                raise IOError('all devices busying')

            print(self.cldevice, " ; ", self.cldevice.version, " ; ", self.cldevice.driver_version)
            # load devicecl.cl
            programFile = open(src_device, 'r')
            programText = programFile.read()
            program     = cl.Program(self.clcontext, programText)
            programFile.close()
            
#            print ('... Build program')
            try:
                program.build()
            except:
                print('-E- error building program')
                print(program.get_build_info(self.cldevice, cl.program_build_info.LOG))
                raise

#            print ('... Load Kernel')
            # load the kernel
            self.kernel = cl.Kernel(program, 'smaccl')
            
        elif exists(binname):
            # load existing binary
            print("....  testsmaccl1 class __init__ exists binary")
            print('Loading binary', binname)
            #self.mod = module_from_buffer(open(binname, 'rb').read())
            # load the kernel
            #self.kernel = self.mod.get_function('Smaccl')
        
        else:
            raise IOError('Could not find {} or {}.'.format(src_device, binname))

        # load the kernel
        #self.kernel = self.mod.get_function('Smaccl')
        
    def copy_to_device(self, name, scalar, dtype):
        
        #print("....  testsmaccl1 class copy_to_device ")
        
        #memcpy_htod(self.mod.get_global(name)[0], np.array([scalar], dtype=dtype))
        dataHost = np.array([scalar], dtype=dtype)
        
        ##### TODO !!!!! return cla.to_device(self.clqueue, dataHost)
        return cl.Buffer(self.clcontext, cl.mem_flags.READ_ONLY, dataHost.nbytes)

    def createOutputArray(self, shp, dtype):
        
        #print("....  testsmaccl1 class createOutputArray ")
        
        outputArray = np.array(shp, dtype=dtype)
        
        #return cl.Buffer(self.clcontext, cl.mem_flags.WRITE_ONLY, outputArray.nbytes)
        return cl.Buffer(self.clcontext, cl.mem_flags.WRITE_ONLY , outputArray.nbytes)  # | cl.mem_flags.COPY_HOST_PTR
        
    def createOutputArrayFromBuffer(self, shp, dtype, buf):
        
        #print("....  testsmaccl1 class createOutputArray ")
        
        #outputArray = np.array(shp, dtype=dtype)
        
        #return cl.Buffer(self.clcontext, cl.mem_flags.WRITE_ONLY, outputArray.nbytes)
        return cl.Buffer(self.clcontext, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=buf)  # 
        

    def run(self, coeffs, tetas, tetav, phis, phiv,
                uh2o, uo3, taup550, pression, rtoa, k1p, k2p,
                iaero, XBLOCK=128, XGRID=128, NBLOOP=1):

#        print("....  testsmaccl1 class run ")
        
        '''
        Run an atmospheric correction run using SMAC

        Arguments:

            - coeffs : array of smac coefficients, its first dimension detrmines the number of bands processed

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

            - k1p :  k1/k0 coefficient ratio of the first RTLS kernel
                     It is 0. for a lambertian surface. Same dimensions as rtoa:
                    float32 arrays of dimension (XBLOCK,XGRID,Z, NB) where Z is 3rd dimension of pixels,
                        and NB is the number of bands

            - k2p :  k2/k0 coefficient ratio of the first RTLS kernel
                     It is 0. for a lambertian surface. Same dimensions as rtoa:
                    float32 arrays of dimension (XBLOCK,XGRID,Z, NB) where Z is 3rd dimension of pixels,
                        and NB is the number of bands

            - XBLOCK and XGRID: control the number of blocks and grid size for
              the GPU execution

            - NBLOOP: number of runs within a thread for the same pixel (should be used for Monte Carlo draws)

        '''
        NBAND = coeffs.shape[0]
        if len(coeffs.shape) == 2:
            nMod = coeffs.shape[1]
        else:
            nMod = 1

        shp = rtoa.shape
        assert shp[0] == NBAND
        if (rtoa.ndim == 4) :
            NZ  = shp[1]
        else : NZ=1

        #output arrays
        rsurf = np.empty_like(rtoa)
        Jrtoa = np.empty_like(rtoa)
        Juo3  = np.empty_like(rtoa)
        Juh2o = np.empty_like(rtoa)
        Jpre  = np.empty_like(rtoa)
        Jtaup = np.empty_like(rtoa)
        
        rsurfd   = self.createOutputArrayFromBuffer(shp, dtype=np.float32, buf=rsurf)
        Jrtoad   = self.createOutputArrayFromBuffer(shp, dtype=np.float32, buf=Jrtoa)
        Juo3d    = self.createOutputArrayFromBuffer(shp, dtype=np.float32, buf=Juo3)
        Juh2od   = self.createOutputArrayFromBuffer(shp, dtype=np.float32, buf=Juh2o)
        Jpred    = self.createOutputArrayFromBuffer(shp, dtype=np.float32, buf=Jpre)
        Jtaupd   = self.createOutputArrayFromBuffer(shp, dtype=np.float32, buf=Jtaup)
        
        
        clcoeffs   = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=coeffs)
        cltetas    = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=tetas)
        cltetav    = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=tetav)
        clphis     = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=phis)
        clphiv     = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=phiv)
        cluh2o     = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=uh2o)
        cluo3      = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=uo3)
        cluh2o     = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=uh2o)
        cltaup550  = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=taup550)
        clpression = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=pression)
        clrtoa     = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=rtoa)
        clk1p      = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=k1p)
        clk2p      = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=k2p)
        claero     = cl.Buffer(self.clcontext, cl.mem_flags.COPY_HOST_PTR, hostbuf=iaero)
        
        print(".... Smaccl: Smaccl  kernel (run) begin  ")

        exec_evt = self.kernel(self.clqueue, (shp[2],shp[3]), None, 
        clcoeffs, 
        cltetas , 
        cltetav , 
        clphis  , 
        clphiv  , 
        cluh2o , 
        cluo3 , 
        cltaup550 , 
        clpression, 
        clrtoa, 
        rsurfd,
        Jrtoad,
        Juo3d,
        Juh2od,
        Jpred,
        Jtaupd,
        clk1p,
        clk2p,
        claero,
        np.int32(nMod),
        np.int32(NBLOOP),
        np.int32(NBAND),
        np.int32(NZ)
        )
        
        cl.enqueue_copy(self.clqueue, rsurf, rsurfd)
        cl.enqueue_copy(self.clqueue, Jrtoa, Jrtoad)
        cl.enqueue_copy(self.clqueue, Juo3, Juo3d)
        cl.enqueue_copy(self.clqueue, Juh2o, Juh2od)
        cl.enqueue_copy(self.clqueue, Jpre, Jpred)
        cl.enqueue_copy(self.clqueue, Jtaup, Jtaupd)
 
        print(".... Smaccl: Smaccl  kernel (run) end  ")
        
        return ( rsurf, Jrtoa, Juo3, Juh2o, Jpre, Jtaup )

    def set_queue(self, xpu='GPU'):
        typedevice = {'CPU':'Portable Computing Language','GPU':'NVIDIA CUDA'}
#        typedevice = {'CPU':'Intel(R) OpenCL'} #turenne
#        typedevice = {'CPU':'Intel(R) OpenCL','GPU':'NVIDIA CUDA'} # mep
        platforms = cl.get_platforms()
        print(platforms)
        for plat in platforms:
            if typedevice[xpu] == plat.name:
                devices = plat.get_devices()
                for device in devices:
                    try:
                        self.cldevice = device
                        self.clcontext = cl.Context([device])
                        self.clqueue = cl.CommandQueue(self.clcontext)

                        return True
                    except:
                        pass

        print('all devices busying')
        return False


