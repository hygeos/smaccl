#include "src/devicecl.h"

__kernel void smaccl(__global coef_atmos *ca, __global float *tetas_, __global float *tetav_, __global float *phis_, __global float *phiv_, 
                     __global float *uh2o_, __global float *uo3_, __global float *taup550_, __global float *pression_, __global float *rtoa_,
                     __global float *rsurf, __global float *Jrtoa, __global float *Juo3, __global float *Juh2o, __global float *Jpre, __global float *Jtaup 
                    , int NBLOOPd, int NBANDd, int NZd)
 
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    int gs0 = get_global_size(0);
    int gs1 = get_global_size(1);
   

    // current thread iudex
    //const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int XXBLOCKd=gs0;
    const int XXGRIDd=gs1;
    const int XNBLOOPd=NBLOOPd;
    const int XNZd=NZd;
    const int XNBANDd=NBANDd;

    const int idx = gid0 * gs1 + gid1;
    const int NRUN=3 + XNBLOOPd; // number of run necessary to compute some Jacobians with finite difference (here 2 Jacobians)
                                // + 1 reference run + number of additional loops for MC
    const int M=XXBLOCKd * XXGRIDd;
    const float dpre = 10.; // absolute perturbation in pressure (hPa) for Jacobian
    const float dtau_rel = 0.1; // relative perturbation in AOT(550) for Jacobian

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

// loop on the 3rd dimension (remaining pixels)
    for (int ip=0; ip<XNZd; ip++) {

// loop on number of run necessary to compute some Jacobians with finite difference
        for (int ir=0; ir<NRUN; ir++) {

// loop on the number of bands
            for (int ib=0; ib<XNBANDd; ib++) {

//
                unsigned long int ii = idx + ip*M + ib*M*XNZd;
                unsigned long int jj = idx + ip*M;
//if (gid0==0) {printf("2.. Processing smaccl !!!!  %08d %08d %08d %08d %08d %08d\n", XNZd, XXGRIDd, XXBLOCKd,XNBANDd,XNBLOOPd,M);
//             printf("... %08d %08d %08d %08d %08d\n", ii, jj, idx, ip, ib);}
//if (gid0<3) printf("... Processing smaccl !!!! %08d %08d %08d %08d %08d %08d %08d %08d %08d %08d %08d\n", ii, jj, idx, ip, ib, M, XNZd, XXGRIDd, XXBLOCKd,XNBANDd,XNBLOOPd);
                float tetas=tetas_[jj], tetav=tetav_[jj], phis=phis_[jj], phiv=phiv_[jj], uh2o=uh2o_[jj], uo3=uo3_[jj]; 
                float taup550=taup550_[jj], pression=pression_[jj], rtoa=rtoa_[ii];
                float dtau = dtau_rel * taup550;
//

                if (ir==0) pression -= dpre;
                if (ir==1) taup550  -= dtau;

                us = cos (tetas*cdr);
                uv = cos (tetav*cdr);
                dphi=(phis-phiv)*cdr;
                Peq=pression/1013.0;
  
                /*------ 1) air mass */
                m =  1./us + 1./uv;

                /*------  2) aerosol optical depth in the spectral band, taup  */
                taup = (ca[ib].a0taup) + (ca[ib].a1taup) * taup550 ;

                /*------  3) gaseous transmissions (downward and upward paths)*/
                to3 = 1. ;
                th2o= 1. ;
                to2 = 1. ;
                tco2= 1. ;
                tch4= 1. ;

                uo2 = pow (Peq , (ca[ib].po2));
                uco2= pow (Peq , (ca[ib].pco2));
                uch4= pow (Peq , (ca[ib].pch4));
                uno2= pow (Peq , (ca[ib].pno2));
                uco = pow (Peq , (ca[ib].pco));

                /*------  4) if uh2o <= 0 and uo3 <= 0 no gaseous absorption is computed*/
                if( (uh2o> 0.) || ( uo3 > 0.) )
                {
                    to3   = exp ( (ca[ib].ao3)  * pow ( (uo3 *m)  , (ca[ib].no3)  ) ) ;
                    th2o  = exp ( (ca[ib].ah2o) * pow ( (uh2o*m)  , (ca[ib].nh2o) ) ) ;
                    to2   = exp ( (ca[ib].ao2)  * pow ( (uo2 *m)  , (ca[ib].no2)  ) ) ;
                    tco2  = exp ( (ca[ib].aco2) * pow ( (uco2*m)  , (ca[ib].nco2) ) ) ;
                    tch4  = exp ( (ca[ib].ach4) * pow ( (uch4*m)  , (ca[ib].nch4) ) ) ;
                    tno2  = exp ( (ca[ib].ano2) * pow ( (uno2*m)  , (ca[ib].nno2) ) ) ;
                    tco   = exp ( (ca[ib].aco)  * pow ( (uco *m)  , (ca[ib].nco)  ) ) ;
                }

                /*------  5) Total scattering transmission */
                ttetas = (ca[ib].a0T) + (ca[ib].a1T)*taup550/us + ((ca[ib].a2T)*Peq + (ca[ib].a3T))/(1.+us) ; /* downward */
                ttetav = (ca[ib].a0T) + (ca[ib].a1T)*taup550/uv + ((ca[ib].a2T)*Peq + (ca[ib].a3T))/(1.+uv) ; /* upward   */

                /*------  6) spherical albedo of the atmosphere */
                s = (ca[ib].a0s) * Peq +  (ca[ib].a3s) + (ca[ib].a1s)*taup550 + (ca[ib].a2s) *pow (taup550 , 2) ;

                /*------  7) scattering angle cosine */
                cksi = - ( (us*uv) + (sqrt(1. - us*us) * sqrt (1. - uv*uv)*cos(dphi) ) );
                if (cksi < -1 ) cksi=-1.0 ;

                /*------  8) scattering angle in degree */
                ksiD = crd*acos(cksi) ;

                /*------  9) rayleigh atmospheric reflectance */
                /* pour 6s on a delta = 0.0279 */
                ray_phase = 0.7190443 * (1. + (cksi*cksi))  + 0.0412742 ;

                taurz=(ca[ib].taur)*Peq;

                ray_ref   = ( taurz*ray_phase ) / (4.*us*uv) ;

                /*-----------------Residu Rayleigh ---------*/
                Res_ray= (ca[ib].Resr1) + (ca[ib].Resr2) * taurz*ray_phase / (us*uv) +
                 (ca[ib].Resr3) * pow( (taurz*ray_phase/(us*uv)),2);

                /*------  10) aerosol atmospheric reflectance */
                aer_phase = (ca[ib].a0P) + (ca[ib].a1P)*ksiD + (ca[ib].a2P)*ksiD*ksiD +(ca[ib].a3P)*pow(ksiD,3) + (ca[ib].a4P) * pow(ksiD,4);

                ak2 = (1. - (ca[ib].wo))*(3. - (ca[ib].wo)*3*(ca[ib].gc)) ;
                ak  = sqrt(ak2) ;
                e   = -3.*us*us*(ca[ib].wo) /  (4.*(1. - ak2*us*us) ) ;
                f   = -(1. - (ca[ib].wo))*3.*(ca[ib].gc)*us*us*(ca[ib].wo) / (4.*(1. - ak2*us*us) ) ;
                dp  = e / (3.*us) + us*f ;
                d   = e + f ;
                b   = 2.*ak / (3. - (ca[ib].wo)*3*(ca[ib].gc));
                del = exp( ak*taup )*(1. + b)*(1. + b) - exp(-ak*taup)*(1. - b)*(1. - b) ;
                ww  = (ca[ib].wo)/4.;
                ss  = us / (1. - ak2*us*us) ;
                q1  = 2. + 3.*us + (1. - (ca[ib].wo))*3.*(ca[ib].gc)*us*(1. + 2.*us) ;
                q2  = 2. - 3.*us - (1. - (ca[ib].wo))*3.*(ca[ib].gc)*us*(1. - 2.*us) ;
                q3  = q2*exp( -taup/us ) ;
                c1  =  ((ww*ss) / del) * ( q1*exp(ak*taup)*(1. + b) + q3*(1. - b) ) ;
                c2  = -((ww*ss) / del) * (q1*exp(-ak*taup)*(1. - b) + q3*(1. + b) ) ;
                cp1 =  c1*ak / ( 3. - (ca[ib].wo)*3.*(ca[ib].gc) ) ;
                cp2 = -c2*ak / ( 3. - (ca[ib].wo)*3.*(ca[ib].gc) ) ;
                z   = d - (ca[ib].wo)*3.*(ca[ib].gc)*uv*dp + (ca[ib].wo)*aer_phase/4. ;
                x   = c1 - (ca[ib].wo)*3.*(ca[ib].gc)*uv*cp1 ;
                y   = c2 - (ca[ib].wo)*3.*(ca[ib].gc)*uv*cp2 ;
                aa1 = uv / (1. + ak*uv) ;
                aa2 = uv / (1. - ak*uv) ;
                aa3 = us*uv / (us + uv) ;

                aer_ref = x*aa1* (1. - exp( -taup/aa1 ) ) ;
                aer_ref = aer_ref + y*aa2*( 1. - exp( -taup / aa2 )  ) ;
                aer_ref = aer_ref + z*aa3*( 1. - exp( -taup / aa3 )  ) ;
                aer_ref = aer_ref / ( us*uv );

                /*--------Residu Aerosol --------*/
                Res_aer= ( (ca[ib].Resa1) + (ca[ib].Resa2) * ( taup * m *cksi ) + (ca[ib].Resa3) * pow( (taup*m*cksi ),2) ) + (ca[ib].Resa4) * pow( (taup*m*cksi),3);


                /*---------Residu 6s-----------*/
                tautot=taup+taurz;
                Res_6s= ( (ca[ib].Rest1) + (ca[ib].Rest2) * ( tautot * m *cksi )
                    + (ca[ib].Rest3) * pow( (tautot*m*cksi),2) ) + (ca[ib].Rest4) * pow( (tautot*m*cksi),3);

                /*------  11) total atmospheric reflectance */
                atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s;

                /*-------- reflectance at toa*/

                tg      = th2o * to3 * to2 * tco2 * tch4* tco * tno2 ;

                 /* reflectance at surface */
                /*------------------------*/
                rsurf[ii] = rtoa - (atm_ref * tg) ;
                rsurf[ii] = rsurf[ii] / ( (tg * ttetas * ttetav) + (rsurf[ii] * s) ) ;
  
                /* Analytical Jacobian of surface reflectance vs toa reflectance*/
                /*------------------------*/
                float ttt   = tg * ttetas * ttetav;
                float rp    = rtoa - atm_ref * tg;
                float rps   = s * rp;
                float delta = 1./(rps + ttt);
                Jrtoa[ii] = delta * ( delta - rps) ;
       
                /* Analytical Jacobian of surface reflectance vs ozone column*/
                /*------------------------*/
                float tgp  = tg/to3;
                float dtdu = (ca[ib].ao3*ca[ib].no3/uo3) * pow ( (uo3 *m) , (ca[ib].no3) ) * to3;
                float drdt = delta * ( (-atm_ref * tgp)  +
                             rp * (atm_ref * s * tg - ttt)/to3 * delta);
                Juo3[ii]  = drdt * dtdu;

                /* Analytical Jacobian of surface reflectance vs water vapour column*/
                /*------------------------*/
                tgp  = tg/th2o;
                dtdu = (ca[ib].ah2o*ca[ib].nh2o/uh2o) * pow ( (uh2o *m) , (ca[ib].nh2o) ) * th2o;
                drdt = delta * ( (-atm_ref * tgp)  +
                             rp * (atm_ref * s * tg - ttt)/th2o * delta);
                Juh2o[ii]  = drdt * dtdu;
  
                /* Finite difference Jacobians of surface reflectance vs pressure and taup550*/
                /*------------------------*/
                if (ir==0) Jpre[ii]  = -rsurf[ii];
                if (ir==1) Jtaup[ii] = -rsurf[ii];
                if (ir==2) {
                    Jpre[ii]  += rsurf[ii];
                    Jpre[ii]  /= dpre;
                    Jtaup[ii] += rsurf[ii];
                    Jtaup[ii] /= dtau;
                }
            } // main loop (ib)
        } // main loop (ir)
    } // main loop (ip)

}

__kernel void smaccl_dir(__global coef_atmos *ca, __global float *tetas_, __global float *tetav_, __global float *phis_, __global float *phiv_, 
                     __global float *uh2o_, __global float *uo3_, __global float *taup550_, __global float *pression_, __global float *rtoa,
                     __global float *rsurf_, __global float *Jrtoa, __global float *Juo3, __global float *Juh2o, __global float *Jpre, __global float *Jtaup 
                    , int NBLOOPd, int NBANDd, int NZd)
 
{
    int gid0 = get_global_id(0);
    int gid1 = get_global_id(1);
    int gs0 = get_global_size(0);
    int gs1 = get_global_size(1);
   

    // current thread iudex
    //const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    const int XXBLOCKd=gs0;
    const int XXGRIDd=gs1;
    const int XNBLOOPd=NBLOOPd;
    const int XNZd=NZd;
    const int XNBANDd=NBANDd;

    const int idx = gid0 * gs1 + gid1;
    const int NRUN=3 + XNBLOOPd; // number of run necessary to compute some Jacobians with finite difference (here 2 Jacobians)
                                // + 1 reference run + number of additional loops for MC
    const int M=XXBLOCKd * XXGRIDd;
    const float dpre = 10.; // absolute perturbation in pressure (hPa) for Jacobian
    const float dtau_rel = 0.1; // relative perturbation in AOT(550) for Jacobian

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

// loop on the 3rd dimension (remaining pixels)
    for (int ip=0; ip<XNZd; ip++) {

// loop on number of run necessary to compute some Jacobians with finite difference
        for (int ir=0; ir<NRUN; ir++) {

// loop on the number of bands
            for (int ib=0; ib<XNBANDd; ib++) {

//
                unsigned long int ii = idx + ip*M + ib*M*XNZd;
                unsigned long int jj = idx + ip*M;
//if (gid0==0) {printf("2.. Processing smaccl !!!!  %08d %08d %08d %08d %08d %08d\n", XNZd, XXGRIDd, XXBLOCKd,XNBANDd,XNBLOOPd,M);
//             printf("... %08d %08d %08d %08d %08d\n", ii, jj, idx, ip, ib);}
//if (gid0<3) printf("... Processing smaccl !!!! %08d %08d %08d %08d %08d %08d %08d %08d %08d %08d %08d\n", ii, jj, idx, ip, ib, M, XNZd, XXGRIDd, XXBLOCKd,XNBANDd,XNBLOOPd);
                float tetas=tetas_[jj], tetav=tetav_[jj], phis=phis_[jj], phiv=phiv_[jj], uh2o=uh2o_[jj], uo3=uo3_[jj]; 
                float taup550=taup550_[jj], pression=pression_[jj], rsurf=rsurf_[ii];
                float dtau = dtau_rel * taup550;
//

                if (ir==0) pression -= dpre;
                if (ir==1) taup550  -= dtau;

                us = cos (tetas*cdr);
                uv = cos (tetav*cdr);
                dphi=(phis-phiv)*cdr;
                Peq=pression/1013.0;
  
                /*------ 1) air mass */
                m =  1./us + 1./uv;

                /*------  2) aerosol optical depth in the spectral band, taup  */
                taup = (ca[ib].a0taup) + (ca[ib].a1taup) * taup550 ;

                /*------  3) gaseous transmissions (downward and upward paths)*/
                to3 = 1. ;
                th2o= 1. ;
                to2 = 1. ;
                tco2= 1. ;
                tch4= 1. ;

                uo2 = pow (Peq , (ca[ib].po2));
                uco2= pow (Peq , (ca[ib].pco2));
                uch4= pow (Peq , (ca[ib].pch4));
                uno2= pow (Peq , (ca[ib].pno2));
                uco = pow (Peq , (ca[ib].pco));

                /*------  4) if uh2o <= 0 and uo3 <= 0 no gaseous absorption is computed*/
                if( (uh2o> 0.) || ( uo3 > 0.) )
                {
                    to3   = exp ( (ca[ib].ao3)  * pow ( (uo3 *m)  , (ca[ib].no3)  ) ) ;
                    th2o  = exp ( (ca[ib].ah2o) * pow ( (uh2o*m)  , (ca[ib].nh2o) ) ) ;
                    to2   = exp ( (ca[ib].ao2)  * pow ( (uo2 *m)  , (ca[ib].no2)  ) ) ;
                    tco2  = exp ( (ca[ib].aco2) * pow ( (uco2*m)  , (ca[ib].nco2) ) ) ;
                    tch4  = exp ( (ca[ib].ach4) * pow ( (uch4*m)  , (ca[ib].nch4) ) ) ;
                    tno2  = exp ( (ca[ib].ano2) * pow ( (uno2*m)  , (ca[ib].nno2) ) ) ;
                    tco   = exp ( (ca[ib].aco)  * pow ( (uco *m)  , (ca[ib].nco)  ) ) ;
                }

                /*------  5) Total scattering transmission */
                ttetas = (ca[ib].a0T) + (ca[ib].a1T)*taup550/us + ((ca[ib].a2T)*Peq + (ca[ib].a3T))/(1.+us) ; /* downward */
                ttetav = (ca[ib].a0T) + (ca[ib].a1T)*taup550/uv + ((ca[ib].a2T)*Peq + (ca[ib].a3T))/(1.+uv) ; /* upward   */

                /*------  6) spherical albedo of the atmosphere */
                s = (ca[ib].a0s) * Peq +  (ca[ib].a3s) + (ca[ib].a1s)*taup550 + (ca[ib].a2s) *pow (taup550 , 2) ;

                /*------  7) scattering angle cosine */
                cksi = - ( (us*uv) + (sqrt(1. - us*us) * sqrt (1. - uv*uv)*cos(dphi) ) );
                if (cksi < -1 ) cksi=-1.0 ;

                /*------  8) scattering angle in degree */
                ksiD = crd*acos(cksi) ;

                /*------  9) rayleigh atmospheric reflectance */
                /* pour 6s on a delta = 0.0279 */
                ray_phase = 0.7190443 * (1. + (cksi*cksi))  + 0.0412742 ;

                taurz=(ca[ib].taur)*Peq;

                ray_ref   = ( taurz*ray_phase ) / (4.*us*uv) ;

                /*-----------------Residu Rayleigh ---------*/
                Res_ray= (ca[ib].Resr1) + (ca[ib].Resr2) * taurz*ray_phase / (us*uv) +
                 (ca[ib].Resr3) * pow( (taurz*ray_phase/(us*uv)),2);

                /*------  10) aerosol atmospheric reflectance */
                aer_phase = (ca[ib].a0P) + (ca[ib].a1P)*ksiD + (ca[ib].a2P)*ksiD*ksiD +(ca[ib].a3P)*pow(ksiD,3) + (ca[ib].a4P) * pow(ksiD,4);

                ak2 = (1. - (ca[ib].wo))*(3. - (ca[ib].wo)*3*(ca[ib].gc)) ;
                ak  = sqrt(ak2) ;
                e   = -3.*us*us*(ca[ib].wo) /  (4.*(1. - ak2*us*us) ) ;
                f   = -(1. - (ca[ib].wo))*3.*(ca[ib].gc)*us*us*(ca[ib].wo) / (4.*(1. - ak2*us*us) ) ;
                dp  = e / (3.*us) + us*f ;
                d   = e + f ;
                b   = 2.*ak / (3. - (ca[ib].wo)*3*(ca[ib].gc));
                del = exp( ak*taup )*(1. + b)*(1. + b) - exp(-ak*taup)*(1. - b)*(1. - b) ;
                ww  = (ca[ib].wo)/4.;
                ss  = us / (1. - ak2*us*us) ;
                q1  = 2. + 3.*us + (1. - (ca[ib].wo))*3.*(ca[ib].gc)*us*(1. + 2.*us) ;
                q2  = 2. - 3.*us - (1. - (ca[ib].wo))*3.*(ca[ib].gc)*us*(1. - 2.*us) ;
                q3  = q2*exp( -taup/us ) ;
                c1  =  ((ww*ss) / del) * ( q1*exp(ak*taup)*(1. + b) + q3*(1. - b) ) ;
                c2  = -((ww*ss) / del) * (q1*exp(-ak*taup)*(1. - b) + q3*(1. + b) ) ;
                cp1 =  c1*ak / ( 3. - (ca[ib].wo)*3.*(ca[ib].gc) ) ;
                cp2 = -c2*ak / ( 3. - (ca[ib].wo)*3.*(ca[ib].gc) ) ;
                z   = d - (ca[ib].wo)*3.*(ca[ib].gc)*uv*dp + (ca[ib].wo)*aer_phase/4. ;
                x   = c1 - (ca[ib].wo)*3.*(ca[ib].gc)*uv*cp1 ;
                y   = c2 - (ca[ib].wo)*3.*(ca[ib].gc)*uv*cp2 ;
                aa1 = uv / (1. + ak*uv) ;
                aa2 = uv / (1. - ak*uv) ;
                aa3 = us*uv / (us + uv) ;

                aer_ref = x*aa1* (1. - exp( -taup/aa1 ) ) ;
                aer_ref = aer_ref + y*aa2*( 1. - exp( -taup / aa2 )  ) ;
                aer_ref = aer_ref + z*aa3*( 1. - exp( -taup / aa3 )  ) ;
                aer_ref = aer_ref / ( us*uv );

                /*--------Residu Aerosol --------*/
                Res_aer= ( (ca[ib].Resa1) + (ca[ib].Resa2) * ( taup * m *cksi ) + (ca[ib].Resa3) * pow( (taup*m*cksi ),2) ) + (ca[ib].Resa4) * pow( (taup*m*cksi),3);


                /*---------Residu 6s-----------*/
                tautot=taup+taurz;
                Res_6s= ( (ca[ib].Rest1) + (ca[ib].Rest2) * ( tautot * m *cksi )
                    + (ca[ib].Rest3) * pow( (tautot*m*cksi),2) ) + (ca[ib].Rest4) * pow( (tautot*m*cksi),3);

                /*------  11) total atmospheric reflectance */
                atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s;

                /*-------- reflectance at toa*/

                tg      = th2o * to3 * to2 * tco2 * tch4* tco * tno2 ;

                 /* reflectance at surface */
                /*------------------------*/
                rtoa[ii] = rsurf * tg * ttetas * ttetav/(1. -rsurf*s) +(atm_ref*tg) ;
  
                /* Analytical Jacobian of surface reflectance vs toa reflectance*/
                /*------------------------*/
                float ttt   = tg * ttetas * ttetav;
                float rp    = rtoa[ii] - atm_ref * tg;
                float rps   = s * rp;
                float delta = 1./(rps + ttt);
                Jrtoa[ii] = delta * ( delta - rps) ;
       
                /* Analytical Jacobian of surface reflectance vs ozone column*/
                /*------------------------*/
                float tgp  = tg/to3;
                float dtdu = (ca[ib].ao3*ca[ib].no3/uo3) * pow ( (uo3 *m) , (ca[ib].no3) ) * to3;
                float drdt = delta * ( (-atm_ref * tgp)  +
                             rp * (atm_ref * s * tg - ttt)/to3 * delta);
                Juo3[ii]  = drdt * dtdu;

                /* Analytical Jacobian of surface reflectance vs water vapour column*/
                /*------------------------*/
                tgp  = tg/th2o;
                dtdu = (ca[ib].ah2o*ca[ib].nh2o/uh2o) * pow ( (uh2o *m) , (ca[ib].nh2o) ) * th2o;
                drdt = delta * ( (-atm_ref * tgp)  +
                             rp * (atm_ref * s * tg - ttt)/th2o * delta);
                Juh2o[ii]  = drdt * dtdu;
  
                /* Finite difference Jacobians of surface reflectance vs pressure and taup550*/
                /*------------------------*/
                if (ir==0) Jpre[ii]  = -rtoa[ii];
                if (ir==1) Jtaup[ii] = -rtoa[ii];
                if (ir==2) {
                    Jpre[ii]  += rtoa[ii];
                    Jpre[ii]  /= dpre;
                    Jtaup[ii] += rtoa[ii];
                    Jtaup[ii] /= dtau;
                }
            } // main loop (ib)
        } // main loop (ir)
    } // main loop (ip)

}


//__kernel void smaccl1(__global coef_atmos *ca, __global float *tetas_, __global float *tetav_, __global float *phis_, __global float *phiv_, 
//                     __global float *uh2o_, __global float *uo3_, __global float *taup550_, __global float *pression_, __global float *rtoa_,
//                     __global float *rsurf, __global float *Jrtoa, __global float *Juo3, __global float *Juh2o, __global float *Jpre, __global float *Jtaup,
//                     int NBLOOPd, int XGRIDd, int XBLOCKd, int NBANDd, int NZd)
// 
//{
////*/
//// current thread iudex
//// TODO: const int idx = threadIdx.x + blockDim.x * blockIdx.x;
//   int gid = get_global_id(0);
//   int gs = get_global_size(0);
//   
//   printf("I am In smaccl !!!! %d / %d :-) \n", gid,gs);
//
//   printf("... %d / %d / %d / %d / %d / %d  :-) \n", gid, NBLOOPd,XGRIDd,XBLOCKd,NBANDd,NZd);
//
//    float a_temp;
//    float b_temp;
//    float c_temp;
//    
//	a_temp = tetas_[gid]; // my a element (by global ref)
//	b_temp = tetav_[gid]; // my b element (by global ref)
//	
//	c_temp = a_temp+b_temp; // sum of my elements
//	
//	rsurf[gid] = c_temp; // store result in global memory
//	Jrtoa[gid] = c_temp;
//	Juo3[gid] = c_temp;
//	Juh2o[gid] = c_temp;
//	Jpre[gid] = c_temp;
//	Jtaup[gid] = c_temp;
//	
//    printf("... %f \n", rsurf[gid]);                    
//                        
//   printf("I am OUT !!!! :-) \n");
//   
//}
//
////}// extern C
//
//__kernel void smaccltest()
// 
//{
//
//   int gid = get_global_id(0);
//   
//   printf("I am In !!!! %d :-) \n", gid);
//
//
//}
