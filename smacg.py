#!/bin/env python

import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
from pycuda.gpuarray import to_gpu, zeros as gpuzeros

XBLOCK=256
XGRID=256
YGRID=1
YBLOCK=1
N=5000

smod = SourceModule("""
#include <pycuda-complex.hpp>
#include <stdio.h>

  typedef struct
  {
   float ah2o, nh2o;
   float ao3,  no3;
   float ao2,  no2,  po2;
   float aco2, nco2, pco2;
   float ach4, nch4, pch4;
   float ano2, nno2, pno2;
   float aco,  nco,  pco;
   float a0u, a1u, a2u ;
   float a0s, a1s, a2s, a3s;
   float a0T, a1T, a2T, a3T;
   float taur,sr;
   float a0taup, a1taup ;
   float wo, gc;
   float a0P,a1P,a2P,a3P;
   float a4P,a5P;
   float Resa1,Resa2;
   float Resa3,Resa4;
   float Resr1, Resr2, Resr3;
   float Rest1,Rest2;
   float Rest3,Rest4;

  } coef_atmos ;

coef_atmos *ca ;

__global__ void smacg(coef_atmos *ca, float *tetasT, float *tetavT, float *phisT, float *phivT, float *uh2oT, float *uo3T, float *taup550T, float *pressionT, float *r_toaT, float *r_surf)

{
// current thread index
const int idx = threadIdx.x + blockDim.x * blockIdx.x;

/* Declarations SMAC */
/*-------------------*/
float cksi;
float s;
float m;
float tg;
float us,uv,dphi;

float crd=180./M_PI;
float cdr=M_PI/180.;

float to3,th2o,to2, tco2;
float tco, tno2,tch4;
float ttetas,ttetav,ksiD;
float atm_ref;

float ak2, ak, e, f, dp, d, b, del, ww, ss, q1, q2, q3, c1, c2, cp1 ;
float cp2, z, x, y, aa1, aa2, aa3 ;

float uo2, uco2, uch4, uco, uno2  ;
float taup,tautot,taurz;
float Peq ;
float Res_ray, Res_aer, Res_6s;
float ray_phase, ray_ref, aer_ref, aer_phase ;

//
float tetas=tetasT[idx], tetav=tetavT[idx], phis=phisT[idx], phiv=phivT[idx], uh2o=uh2oT[idx], uo3=uo3T[idx]; 
float taup550=taup550T[idx], pression=pressionT[idx], r_toa=r_toaT[idx];
//

for (int ib=0; ib<1000; ib++) {

us = cos (tetas*cdr);
uv = cos (tetav*cdr);
dphi=(phis-phiv)*cdr;
Peq=pression/1013.0;


/*------ 1) air mass */
m =  1./us + 1./uv;

/*------  2) aerosol optical depth in the spectral band, taup  */
taup = (ca->a0taup) + (ca->a1taup) * taup550 ;

/*------  3) gaseous transmissions (downward and upward paths)*/
to3 = 1. ;
th2o= 1. ;
to2 = 1. ;
tco2= 1. ;
tch4= 1. ;

uo2= pow (Peq , (ca->po2));
uco2= pow (Peq , (ca->pco2));
uch4= pow (Peq , (ca->pch4));
uno2= pow (Peq , (ca->pno2));
uco = pow (Peq , (ca->pco));

/*------  4) if uh2o <= 0 and uo3 <= 0 no gaseous absorption is computed*/
if( (uh2o> 0.) || ( uo3 > 0.) )
{
        to3   = exp ( (ca->ao3)  * pow ( (uo3 *m)  , (ca->no3)  ) ) ;
        th2o  = exp ( (ca->ah2o) * pow ( (uh2o*m)  , (ca->nh2o) ) ) ;
        to2   = exp ( (ca->ao2)  * pow ( (uo2 *m)  , (ca->no2)  ) ) ;
        tco2  = exp ( (ca->aco2) * pow ( (uco2*m)  , (ca->nco2) ) ) ;
        tch4  = exp ( (ca->ach4) * pow ( (uch4*m)  , (ca->nch4) ) ) ;
        tno2  = exp ( (ca->ano2) * pow ( (uno2*m)  , (ca->nno2) ) ) ;
        tco   = exp ( (ca->aco)  * pow ( (uco*m)   , (ca->nco) ) ) ;
}

/*------  5) Total scattering transmission */
ttetas = (ca->a0T) + (ca->a1T)*taup550/us + ((ca->a2T)*Peq + (ca->a3T))/(1.+us) ; /* downward */
ttetav = (ca->a0T) + (ca->a1T)*taup550/uv + ((ca->a2T)*Peq + (ca->a3T))/(1.+uv) ; /* upward   */

/*------  6) spherical albedo of the atmosphere */
s = (ca->a0s) * Peq +  (ca->a3s) + (ca->a1s)*taup550 + (ca->a2s) *pow (taup550 , 2) ;

/*------  7) scattering angle cosine */
cksi = - ( (us*uv) + (sqrt(1. - us*us) * sqrt (1. - uv*uv)*cos(dphi) ) );
if (cksi < -1 ) cksi=-1.0 ;

/*------  8) scattering angle in degree */
ksiD = crd*acos(cksi) ;

/*------  9) rayleigh atmospheric reflectance */
/* pour 6s on a delta = 0.0279 */
ray_phase = 0.7190443 * (1. + (cksi*cksi))  + 0.0412742 ;

taurz=(ca->taur)*Peq;

ray_ref   = ( taurz*ray_phase ) / (4.*us*uv) ;

/*-----------------Residu Rayleigh ---------*/
Res_ray= (ca->Resr1) + (ca->Resr2) * taurz*ray_phase / (us*uv) +
         (ca->Resr3) * pow( (taurz*ray_phase/(us*uv)),2);

/*------  10) aerosol atmospheric reflectance */
aer_phase = (ca->a0P) + (ca->a1P)*ksiD + (ca->a2P)*ksiD*ksiD +(ca->a3P)*pow(ksiD,3) + (ca->a4P) * pow(ksiD,4);

ak2 = (1. - (ca->wo))*(3. - (ca->wo)*3*(ca->gc)) ;
ak  = sqrt(ak2) ;
e   = -3.*us*us*(ca->wo) /  (4.*(1. - ak2*us*us) ) ;
f   = -(1. - (ca->wo))*3.*(ca->gc)*us*us*(ca->wo) / (4.*(1. - ak2*us*us) ) ;
dp  = e / (3.*us) + us*f ;
d   = e + f ;
b   = 2.*ak / (3. - (ca->wo)*3*(ca->gc));
del = exp( ak*taup )*(1. + b)*(1. + b) - exp(-ak*taup)*(1. - b)*(1. - b) ;
ww  = (ca->wo)/4.;
ss  = us / (1. - ak2*us*us) ;
q1  = 2. + 3.*us + (1. - (ca->wo))*3.*(ca->gc)*us*(1. + 2.*us) ;
q2  = 2. - 3.*us - (1. - (ca->wo))*3.*(ca->gc)*us*(1. - 2.*us) ;
q3  = q2*exp( -taup/us ) ;
c1  =  ((ww*ss) / del) * ( q1*exp(ak*taup)*(1. + b) + q3*(1. - b) ) ;
c2  = -((ww*ss) / del) * (q1*exp(-ak*taup)*(1. - b) + q3*(1. + b) ) ;
cp1 =  c1*ak / ( 3. - (ca->wo)*3.*(ca->gc) ) ;
cp2 = -c2*ak / ( 3. - (ca->wo)*3.*(ca->gc) ) ;
z   = d - (ca->wo)*3.*(ca->gc)*uv*dp + (ca->wo)*aer_phase/4. ;
x   = c1 - (ca->wo)*3.*(ca->gc)*uv*cp1 ;
y   = c2 - (ca->wo)*3.*(ca->gc)*uv*cp2 ;
aa1 = uv / (1. + ak*uv) ;
aa2 = uv / (1. - ak*uv) ;
aa3 = us*uv / (us + uv) ;

aer_ref = x*aa1* (1. - exp( -taup/aa1 ) ) ;
aer_ref = aer_ref + y*aa2*( 1. - exp( -taup / aa2 )  ) ;
aer_ref = aer_ref + z*aa3*( 1. - exp( -taup / aa3 )  ) ;
aer_ref = aer_ref / ( us*uv );

/*--------Residu Aerosol --------*/
Res_aer= ( (ca->Resa1) + (ca->Resa2) * ( taup * m *cksi ) + (ca->Resa3) * pow( (taup*m*cksi ),2) ) + (ca->Resa4) * pow( (taup*m*cksi),3);


/*---------Residu 6s-----------*/
tautot=taup+taurz;
Res_6s= ( (ca->Rest1) + (ca->Rest2) * ( tautot * m *cksi )
        + (ca->Rest3) * pow( (tautot*m*cksi),2) ) + (ca->Rest4) * pow( (tautot*m*cksi),3);

/*------  11) total atmospheric reflectance */
atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s;

/*-------- reflectance at toa*/

tg      = th2o * to3 * to2 * tco2 * tch4* tco * tno2 ;

 /* reflectance at surface */
/*------------------------*/
  r_surf[idx] = r_toa - (atm_ref * tg) ;
  r_surf[idx] = r_surf[idx] / ( (tg * ttetas * ttetav) + (r_surf[idx] * s) ) ;

} // main loop

}""")

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

if __name__=="__main__" :
    #lecture des coeffs
    coeffs=coeff('COEFFS/coef_NOAA14VIS_CONT.dat')
    smacg_run = smod.get_function("smacg")
    coeff1 = np.zeros((), dtype=type_coeff)

    for k in coeffs.__dict__.keys():
        coeff1[k] = coeffs.__dict__[k]

    # Inputs arrays

    tetas = np.linspace(30.,35.,num=N, dtype=np.float32)
    tetav = np.linspace(0.,50.,num=N, dtype=np.float32)
    phis  = np.zeros(N, dtype=np.float32)
    phiv  = np.linspace(0.,360.,num=N, dtype=np.float32)
    uh2o  = np.random.random(size=N)
    uo3   = np.random.random(size=N)
    taup550   = np.random.random(size=N)
    pression   = np.random.random(size=N)* 50. + 975.
    r_toa = np.linspace(0.1,0.15,num=N, dtype=np.float32)
    r_surf  = gpuzeros(N, dtype=np.float32)

    # run
    smacg_run( to_gpu(coeff1), 
             to_gpu(tetas) , 
             to_gpu(tetav) , 
             to_gpu(phis)  , 
             to_gpu(phiv)  , 
             to_gpu(uh2o.astype(np.float32)) , 
             to_gpu(uo3.astype(np.float32)) , 
             to_gpu(taup550.astype(np.float32)) , 
             to_gpu(pression.astype(np.float32)), 
             to_gpu(r_toa), 
             r_surf, 
             block=(XBLOCK, 1, 1), grid=(XGRID, 1, 1)
            ) 

