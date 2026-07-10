//#include "kernels/devicecl.h"

#define PI 3.1415927F
#define DEUXPI 6.2831853F
#define DEMIPI 1.5707963F
/**********************************************************
*
*           deviceocl.h
*
***********************************************************/

typedef struct
  {
   float ah2o, nh2o;
   float ao3,  no3;
   float ao2,  no2,  po2;
   float aco2, nco2, pco2;
   float ach4, nch4, pch4;
   float ano2, nno2, pno2;
   float aco,  nco,  pco;
   float a0s, a1s, a2s, a3s;
   float a0T, a1T, a2T, a3T;
   float taur;
   float a0taup, a1taup ;
   float wo, gc;
   float a0P,a1P,a2P,a3P;
   float a4P;
   float Resa1,Resa2;
   float Resa3,Resa4;
   float Resr1, Resr2, Resr3;
   float Rest1,Rest2;
   float Rest3,Rest4;
   float f1d0, f1d1, f1d2;
   float f2d0, f2d1, f2d2;
   float f1b0, f1b1, f1b2;
   float f2b0, f2b1, f2b2;

  } coef_atmos ;

/*########## Ross Thick Li-Sparse  ##############*/

float F1_rtls(float ths, float thv, float phi ){  //  rossthick-lisparse, only F1
    if (phi < 0.) phi += DEUXPI; 
    if (phi > PI) phi = DEUXPI - phi; 
    float cos_xi = cos(ths) * cos(thv) + sin(ths) * sin(thv) * cos(phi);
    float mm = 1./cos(thv) + 1./cos(ths);
    float tthv = tan(thv);
    float tths = tan(ths);

    float cos_t = 2./mm * sqrt(tthv*tthv + tths*tths - 2.*tthv*tths * cos(phi) + pow(tthv *tths * sin(phi),2) );
    cos_t = fmin(cos_t, 1.F);
    float t = acos(cos_t);
    float sin_t = sin(t);
    float big_O = mm*(t - sin_t*cos_t)/PI;
            
    // geometric kernel
    float F1 = big_O-(1./cos(thv) + 1./cos(ths)) + (1. + cos_xi)/(cos(thv)*cos(ths))/2.;

    return F1;
}


float F2_rtls(float ths, float thv, float phi ){  //  rossthick-lisparse, only F2
    if (phi < 0.) phi += DEUXPI; 
    if (phi > PI) phi = DEUXPI - phi; 
    float cos_xi = cos(ths) * cos(thv) + sin(ths) * sin(thv) * cos(phi);
    float xi = acos(cos_xi);

    // volume-scattering kernel
    float F2 = (((PI/2. -xi)*cos_xi + sin(xi))/(cos(thv) + cos(ths))) - PI/4.;

    return F2;
}

__kernel void smaccl(__global coef_atmos *ca, __global float *tetas_, __global float *tetav_, __global float *phis_, __global float *phiv_, 
                     __global float *uh2o_, __global float *uo3_, __global float *taup550_, __global float *pression_, __global float *rtoa_,
                     __global float *rsurf, __global float *rsurf_0, __global float *dev_std, __global float *Jr, __global float *Juo3, __global float *Juh2o, __global float *Jpre, 
                     __global float *Jtaup, 
                     __global float *k1p_, __global float *k2p_,
                     __global short *iaero, int NMOD, 
                    int NBLOOPd, int NBANDd, int NZd, int Naerod)
 
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
    const int XNaerod=Naerod;

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
    float tdirtetas,tdirtetav,tdiftetas,tdiftetav,trans_atm,trans_atm_0;
    float atm_ref;

    float ak2, ak, e, f, dp, d, b, del, ww, ss, q1, q2, q3, c1, c2, cp1 ;
    float cp2, z, x, y, aa1, aa2, aa3 ;

    float uo2, uco2, uch4, uco, uno2  ;
    float taup,tautot,taurz;
    float Peq ;
    float Res_ray, Res_aer, Res_6s;
    float ray_phase, ray_ref, aer_ref, aer_phase ;
    short iAero = 0;
    float toc, toc_mean, toc_std, toc_0;

    unsigned long int irt;

    // loop on the 3rd dimension (remaining pixels)
    for (int ip=0; ip<XNZd; ip++) {
        unsigned long int jj = idx + ip*M; 
        float tetas=tetas_[jj], tetav=tetav_[jj], phis=phis_[jj], phiv=phiv_[jj], uh2o=uh2o_[jj], uo3=uo3_[jj]; 
        us = cos (tetas*cdr);
        uv = cos (tetav*cdr);
        dphi=(phis-phiv)*cdr;
        /*------ 1) air mass */
        m =  1./us + 1./uv;

        /*------  7) scattering angle cosine */
        cksi = - ( (us*uv) + (sqrt(1. - us*us) * sqrt (1. - uv*uv)*cos(dphi) ) );
        if (cksi < -1 ) cksi=-1.0 ;

        /*------  8) scattering angle in degree */
        ksiD = crd*acos(cksi) ;

        /*------  9) rayleigh atmospheric reflectance */
        /* pour 6s on a delta = 0.0279 */
        ray_phase = 0.7190443 * (1. + (cksi*cksi))  + 0.0412742 ;

            // loop on the number of bands
        for (int ib=0; ib<XNBANDd; ib++) {
            unsigned long int ii = idx + ip*M + ib*M*XNZd;
            float rtoa=rtoa_[ii];
            float k1p = k1p_[ii];
            float k2p = k2p_[ii];

            toc_mean = 0;
            toc_std = 0;
            for (int ia=0; ia<XNaerod; ia++) {
//                irt = M*XNZd*XNBANDd*ia + ii;
            //    irt = idx + ip*M + ib*M*XNZd + ia*M*XNZd*XNBANDd;
                iAero = iaero[jj + ia*M*XNZd];
                // loop on number of run necessary to compute some Jacobians with finite difference
                for (int ir=0; ir<NRUN; ir++) {
//                    ir = ia + XNaerod*ii;
                    unsigned long int kk = ib*NMOD+iAero;
                    float taup550=taup550_[jj], pression=pression_[jj];

                    float dtau = dtau_rel * taup550;
                    if (ir==0) pression -= dpre;
                    if (ir==1) taup550  -= dtau;
                    Peq=pression/1013.0;
      
                    /*------  2) aerosol optical depth in the spectral band, taup  */
                    taup = (ca[kk].a0taup) + (ca[kk].a1taup) * taup550 ;

                    /*------  3) gaseous transmissions (downward and upward paths)*/
                    to2 = 1. ;
                    tco2= 1. ;
                    tch4= 1. ;
                    tno2= 1. ;
                    tco = 1. ;
                    to3 = 1. ;
                    th2o= 1. ;

                    uo2 = pow (Peq , (ca[kk].po2));
                    uco2= pow (Peq , (ca[kk].pco2));
                    uch4= pow (Peq , (ca[kk].pch4));
                    uno2= pow (Peq , (ca[kk].pno2));
                    uco = pow (Peq , (ca[kk].pco));

                    /*------  4) if uh2o <= 0 and uo3 <= 0 no gaseous absorption is computed*/
                    if( (uh2o> 0.) || ( uo3 > 0.) )
                    {
                        to3   = exp ( (ca[kk].ao3)  * pow ( (uo3 *m)  , (ca[kk].no3)  ) ) ;
                        th2o  = exp ( (ca[kk].ah2o) * pow ( (uh2o*m)  , (ca[kk].nh2o) ) ) ;
                        to2   = exp ( (ca[kk].ao2)  * pow ( (uo2 *m)  , (ca[kk].no2)  ) ) ;
                        tco2  = exp ( (ca[kk].aco2) * pow ( (uco2*m)  , (ca[kk].nco2) ) ) ;
                        tch4  = exp ( (ca[kk].ach4) * pow ( (uch4*m)  , (ca[kk].nch4) ) ) ;
                        tno2  = exp ( (ca[kk].ano2) * pow ( (uno2*m)  , (ca[kk].nno2) ) ) ;
                        tco   = exp ( (ca[kk].aco)  * pow ( (uco *m)  , (ca[kk].nco)  ) ) ;
                    }

                    /*------  5) Total scattering transmission */
                    ttetas = (ca[kk].a0T) + (ca[kk].a1T)*taup550/us + ((ca[kk].a2T)*Peq + (ca[kk].a3T))/(1.+us) ; /* downward */
                    ttetav = (ca[kk].a0T) + (ca[kk].a1T)*taup550/uv + ((ca[kk].a2T)*Peq + (ca[kk].a3T))/(1.+uv) ; /* upward   */

                    /*------  6) spherical albedo of the atmosphere */
                    s = (ca[kk].a0s) * Peq +  (ca[kk].a3s) + (ca[kk].a1s)*taup550 + (ca[kk].a2s) *pow (taup550 , 2) ;

//                    /*------  7) scattering angle cosine */
//                    cksi = - ( (us*uv) + (sqrt(1. - us*us) * sqrt (1. - uv*uv)*cos(dphi) ) );
//                    if (cksi < -1 ) cksi=-1.0 ;
//
//                    /*------  8) scattering angle in degree */
//                    ksiD = crd*acos(cksi) ;
//
//                    /*------  9) rayleigh atmospheric reflectance */
//                    /* pour 6s on a delta = 0.0279 */
//                    ray_phase = 0.7190443 * (1. + (cksi*cksi))  + 0.0412742 ;

                    taurz=(ca[kk].taur)*Peq;

                    ray_ref   = ( taurz*ray_phase ) / (4.*us*uv) ;

                    /*-----------------Residu Rayleigh ---------*/
                    Res_ray= (ca[kk].Resr1) + (ca[kk].Resr2) * taurz*ray_phase / (us*uv) +
                     (ca[kk].Resr3) * pow( (taurz*ray_phase/(us*uv)),2);

                    /*--------9b) Direct and scattering transmssions*/
                    tautot=taup+taurz;
                    tdirtetas = exp(-tautot/us);
                    tdirtetav = exp(-tautot/uv);
                    tdiftetas = ttetas - tdirtetas;
                    tdiftetav = ttetav - tdirtetav;

                    /*------  10) aerosol atmospheric reflectance */
                    aer_phase = (ca[kk].a0P) + (ca[kk].a1P)*ksiD + (ca[kk].a2P)*ksiD*ksiD +(ca[kk].a3P)*pow(ksiD,3) + (ca[kk].a4P) * pow(ksiD,4);

                    ak2 = (1. - (ca[kk].wo))*(3. - (ca[kk].wo)*3*(ca[kk].gc)) ;
                    ak  = sqrt(ak2) ;
                    e   = -3.*us*us*(ca[kk].wo) /  (4.*(1. - ak2*us*us) ) ;
                    f   = -(1. - (ca[kk].wo))*3.*(ca[kk].gc)*us*us*(ca[kk].wo) / (4.*(1. - ak2*us*us) ) ;
                    dp  = e / (3.*us) + us*f ;
                    d   = e + f ;
                    b   = 2.*ak / (3. - (ca[kk].wo)*3*(ca[kk].gc));
                    del = exp( ak*taup )*(1. + b)*(1. + b) - exp(-ak*taup)*(1. - b)*(1. - b) ;
                    ww  = (ca[kk].wo)/4.;
                    ss  = us / (1. - ak2*us*us) ;
                    q1  = 2. + 3.*us + (1. - (ca[kk].wo))*3.*(ca[kk].gc)*us*(1. + 2.*us) ;
                    q2  = 2. - 3.*us - (1. - (ca[kk].wo))*3.*(ca[kk].gc)*us*(1. - 2.*us) ;
                    q3  = q2*exp( -taup/us ) ;
                    c1  =  ((ww*ss) / del) * ( q1*exp(ak*taup)*(1. + b) + q3*(1. - b) ) ;
                    c2  = -((ww*ss) / del) * (q1*exp(-ak*taup)*(1. - b) + q3*(1. + b) ) ;
                    cp1 =  c1*ak / ( 3. - (ca[kk].wo)*3.*(ca[kk].gc) ) ;
                    cp2 = -c2*ak / ( 3. - (ca[kk].wo)*3.*(ca[kk].gc) ) ;
                    z   = d - (ca[kk].wo)*3.*(ca[kk].gc)*uv*dp + (ca[kk].wo)*aer_phase/4. ;
                    x   = c1 - (ca[kk].wo)*3.*(ca[kk].gc)*uv*cp1 ;
                    y   = c2 - (ca[kk].wo)*3.*(ca[kk].gc)*uv*cp2 ;
                    aa1 = uv / (1. + ak*uv) ;
                    aa2 = uv / (1. - ak*uv) ;
                    aa3 = us*uv / (us + uv) ;

                    aer_ref = x*aa1* (1. - exp( -taup/aa1 ) ) ;
                    aer_ref = aer_ref + y*aa2*( 1. - exp( -taup / aa2 )  ) ;
                    aer_ref = aer_ref + z*aa3*( 1. - exp( -taup / aa3 )  ) ;
                    aer_ref = aer_ref / ( us*uv );

                    /*--------Residu Aerosol --------*/
                    Res_aer= ( (ca[kk].Resa1) + (ca[kk].Resa2) * ( taup * m *cksi ) + (ca[kk].Resa3) * pow( (taup*m*cksi ),2) ) 
                             + (ca[kk].Resa4) * pow( (taup*m*cksi),3);


                    /*---------Residu 6s-----------*/
                    Res_6s= ( (ca[kk].Rest1) + (ca[kk].Rest2) * ( tautot * m *cksi )
                        + (ca[kk].Rest3) * pow( (tautot*m*cksi),2) ) + (ca[kk].Rest4) * pow( (tautot*m*cksi),3);

                    /*------  11) total atmospheric reflectance */
                    atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s;

                    /*-------- reflectance at toa*/

                    tg      = th2o * to3 * to2 * tco2 * tch4* tco * tno2 ;

                     /* reflectance at surface */
                    /*------------------------*/
//                    rsurf[irt] = rtoa - (atm_ref * tg) ;
                    toc = rtoa - (atm_ref * tg) ;
                    float ax1 = F1_rtls(tetas*cdr, tetav*cdr, dphi);
                    float ax2 = F2_rtls(tetas*cdr, tetav*cdr, dphi);
                    float f1_bar_down = ca[kk].f1d0 + ca[kk].f1d1*ax1 + ca[kk].f1d2*ax2; 
                    float f2_bar_down = ca[kk].f2d0 + ca[kk].f2d1*ax1 + ca[kk].f2d2*ax2; 
                    float f1_bar_bar  = ca[kk].f1b0 + ca[kk].f1b1*ax1 + ca[kk].f1b2*ax2; 
                    float f2_bar_bar  = ca[kk].f2b0 + ca[kk].f2b1*ax1 + ca[kk].f2b2*ax2; 
                    float rs          =  (1. + k1p*ax1 + k2p*ax2);
                    ax1 = F1_rtls(tetav*cdr, tetas*cdr, M_PI-dphi);
                    ax2 = F2_rtls(tetav*cdr, tetas*cdr, M_PI-dphi);
                    float f1_bar_up   = ca[kk].f1d0 + ca[kk].f1d1*ax1 + ca[kk].f1d2*ax2; 
                    float f2_bar_up   = ca[kk].f2d0 + ca[kk].f2d1*ax1 + ca[kk].f2d2*ax2; 
                    /**/
                    float ref_surf_bar_downN= (1. + k1p*f1_bar_down + k2p*f2_bar_down)/rs;
                    float ref_surf_bar_upN  = (1. + k1p*f1_bar_up   + k2p*f2_bar_up)  /rs;
                    float ref_surf_bar_barN = (1. + k1p*f1_bar_bar  + k2p*f2_bar_bar )/rs;
                    trans_atm = (tdirtetav*tdirtetas) +
                                (tdirtetav*tdiftetas) * (ref_surf_bar_downN) +
                                (tdiftetav*tdirtetas) * (ref_surf_bar_upN) +
                                (tdiftetav*tdiftetas) * (ref_surf_bar_barN);
                    trans_atm_0 = (tdirtetav*tdirtetas) + (tdirtetav*tdiftetas) + (tdiftetav*tdirtetas) + (tdiftetav*tdiftetas);
//                    rsurf[irt] = rsurf[irt] / ( (tg * trans_atm) + (rsurf[irt] * s) ) ;
                    toc_0 = toc / ( (tg * trans_atm_0) + (toc * s) ) ;
                    toc = toc / ( (tg * trans_atm) + (toc * s) ) ;
                    
                    if (ia==0) {
                        rsurf[ii] = toc;
                        rsurf_0[ii] = toc_0;
                        /* Analytical Jacobian of surface reflectance vs toa reflectance*/
                        /*------------------------*/
                        float ttt   = tg * trans_atm;
                        //float ttt   = tg * ttetas * ttetav;
                        float rp    = rtoa - atm_ref * tg;
                        float rps   = s * rp;
                        //float delta = 1./(rps + ttt);
                        float nu_inv = 1./(rps + ttt);
                        Jr[ii] = nu_inv *  nu_inv * ttt ;
               
                        /* Analytical Jacobian of surface reflectance vs ozone column*/
                        /*------------------------*/
                        float tgp  = tg/to3;
                        float dtdu = (ca[kk].ao3*ca[kk].no3/uo3) * pow ( (uo3 *m) , (ca[kk].no3) ) * to3;
                        //float drdt = delta * ( (-atm_ref * tgp)  +
                        // rp * (atm_ref * s * tg - ttt)/to3 * delta);
                        float drdt =  -rtoa * nu_inv*nu_inv * ttt/to3;
                        Juo3[ii]  = drdt * dtdu;

                        /* Analytical Jacobian of surface reflectance vs water vapour column*/
                        /*------------------------*/
                        tgp  = tg/th2o;
                        dtdu = (ca[kk].ah2o*ca[kk].nh2o/uh2o) * pow ( (uh2o *m) , (ca[kk].nh2o) ) * th2o;
                        //drdt = delta * ( (-atm_ref * tgp)  +
                        //            rp * (atm_ref * s * tg - ttt)/th2o * delta);
                        drdt =  -rtoa * nu_inv*nu_inv * ttt/th2o;
                        Juh2o[ii]  = drdt * dtdu;
          
                        /* Finite difference Jacobians of surface reflectance vs pressure and taup550*/
                        /*------------------------*/
//                        if (ir==0) Jpre[ii]  = -rsurf[irt];
//                        if (ir==1) Jtaup[ii] = -rsurf[irt];
//                        if (ir==2) {
//                            Jpre[ii]  += rsurf[irt];
//                            Jtaup[ii] += rsurf[irt];
//                        }
                        if (ir==0) Jpre[ii]  = -rsurf[ii];
                        if (ir==1) Jtaup[ii] = -rsurf[ii];
                        if (ir==2) {
                            Jpre[ii]  += rsurf[ii];
                            Jpre[ii]  /= dpre;
                            Jtaup[ii] += rsurf[ii];
                            Jtaup[ii] /= dtau;
                        }
                    }
                } // main loop (ir)
                if (ia!=0) {
                    toc_mean += toc;
                    toc_std  += toc*toc;
                }
            } // main loop (ia)
            toc_mean /= (float)(XNaerod -1);
            toc_std /= (float)(XNaerod -1);
            dev_std[ii] = sqrt( fabsf(toc_std - toc_mean*toc_mean ));
        } // main loop (ib)
    } // main loop (ip)

}


__kernel void smaccl_dir(__global coef_atmos *ca, __global float *tetas_, __global float *tetav_, __global float *phis_, __global float *phiv_, 
                     __global float *uh2o_, __global float *uo3_, __global float *taup550_, __global float *pression_, __global float *rtoa,
                     __global float *rsurf_, __global float *Jr, __global float *Juo3, __global float *Juh2o, __global float *Jpre, 
                     __global float *Jtaup, 
                     __global float *k1p_, __global float *k2p_,
                     __global int *iaero, int NMOD, 
                    int NBLOOPd, int NBANDd, int NZd)
 
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
    float tdirtetas,tdirtetav,tdiftetas,tdiftetav,trans_atm;
    float atm_ref;

    float ak2, ak, e, f, dp, d, b, del, ww, ss, q1, q2, q3, c1, c2, cp1 ;
    float cp2, z, x, y, aa1, aa2, aa3 ;

    float uo2, uco2, uch4, uco, uno2  ;
    float taup,tautot,taurz;
    float Peq ;
    float Res_ray, Res_aer, Res_6s;
    float ray_phase, ray_ref, aer_ref, aer_phase ;
    int iAero = 0;

    // loop on the 3rd dimension (remaining pixels)
    for (int ip=0; ip<XNZd; ip++) {
        unsigned long int jj = idx + ip*M;
        iAero = iaero[jj];
        float tetas=tetas_[jj], tetav=tetav_[jj], phis=phis_[jj], phiv=phiv_[jj], uh2o=uh2o_[jj], uo3=uo3_[jj]; 

         // loop on number of run necessary to compute some Jacobians with finite difference
        for (int ir=0; ir<NRUN; ir++) {

            // loop on the number of bands
            for (int ib=0; ib<XNBANDd; ib++) {
                unsigned long int ii = idx + ip*M + ib*M*XNZd;
                unsigned long int kk = ib*NMOD+iAero;
                float taup550=taup550_[jj], pression=pression_[jj];
                float rsurf=rsurf_[ii];
                //float rtoa=rtoa_[ii];
                float k1p = k1p_[ii];
                float k2p = k2p_[ii];

                float dtau = dtau_rel * taup550;
                if (ir==0) pression -= dpre;
                if (ir==1) taup550  -= dtau;

                us = cos (tetas*cdr);
                uv = cos (tetav*cdr);
                dphi=(phis-phiv)*cdr;
                Peq=pression/1013.0;
  
                /*------ 1) air mass */
                m =  1./us + 1./uv;

                /*------  2) aerosol optical depth in the spectral band, taup  */
                taup = (ca[kk].a0taup) + (ca[kk].a1taup) * taup550 ;

                /*------  3) gaseous transmissions (downward and upward paths)*/
                to3 = 1. ;
                th2o= 1. ;
                to2 = 1. ;
                tco2= 1. ;
                tch4= 1. ;

                uo2 = pow (Peq , (ca[kk].po2));
                uco2= pow (Peq , (ca[kk].pco2));
                uch4= pow (Peq , (ca[kk].pch4));
                uno2= pow (Peq , (ca[kk].pno2));
                uco = pow (Peq , (ca[kk].pco));

                /*------  4) if uh2o <= 0 and uo3 <= 0 no gaseous absorption is computed*/
                if( (uh2o> 0.) || ( uo3 > 0.) )
                {
                    to3   = exp ( (ca[kk].ao3)  * pow ( (uo3 *m)  , (ca[kk].no3)  ) ) ;
                    th2o  = exp ( (ca[kk].ah2o) * pow ( (uh2o*m)  , (ca[kk].nh2o) ) ) ;
                    to2   = exp ( (ca[kk].ao2)  * pow ( (uo2 *m)  , (ca[kk].no2)  ) ) ;
                    tco2  = exp ( (ca[kk].aco2) * pow ( (uco2*m)  , (ca[kk].nco2) ) ) ;
                    tch4  = exp ( (ca[kk].ach4) * pow ( (uch4*m)  , (ca[kk].nch4) ) ) ;
                    tno2  = exp ( (ca[kk].ano2) * pow ( (uno2*m)  , (ca[kk].nno2) ) ) ;
                    tco   = exp ( (ca[kk].aco)  * pow ( (uco *m)  , (ca[kk].nco)  ) ) ;
                }

                /*------  5) Total scattering transmission */
                ttetas = (ca[kk].a0T) + (ca[kk].a1T)*taup550/us + ((ca[kk].a2T)*Peq + (ca[kk].a3T))/(1.+us) ; /* downward */
                ttetav = (ca[kk].a0T) + (ca[kk].a1T)*taup550/uv + ((ca[kk].a2T)*Peq + (ca[kk].a3T))/(1.+uv) ; /* upward   */

                /*------  6) spherical albedo of the atmosphere */
                s = (ca[kk].a0s) * Peq +  (ca[kk].a3s) + (ca[kk].a1s)*taup550 + (ca[kk].a2s) *pow (taup550 , 2) ;

                /*------  7) scattering angle cosine */
                cksi = - ( (us*uv) + (sqrt(1. - us*us) * sqrt (1. - uv*uv)*cos(dphi) ) );
                if (cksi < -1 ) cksi=-1.0 ;

                /*------  8) scattering angle in degree */
                ksiD = crd*acos(cksi) ;

                /*------  9) rayleigh atmospheric reflectance */
                /* pour 6s on a delta = 0.0279 */
                ray_phase = 0.7190443 * (1. + (cksi*cksi))  + 0.0412742 ;

                taurz=(ca[kk].taur)*Peq;

                ray_ref   = ( taurz*ray_phase ) / (4.*us*uv) ;

                /*-----------------Residu Rayleigh ---------*/
                Res_ray= (ca[kk].Resr1) + (ca[kk].Resr2) * taurz*ray_phase / (us*uv) +
                 (ca[kk].Resr3) * pow( (taurz*ray_phase/(us*uv)),2);

                /*--------9b) Direct and scattering transmssions*/
                tautot=taup+taurz;
                tdirtetas = exp(-tautot/us);
                tdirtetav = exp(-tautot/uv);
                tdiftetas = ttetas - tdirtetas;
                tdiftetav = ttetav - tdirtetav;

                /*------  10) aerosol atmospheric reflectance */
                aer_phase = (ca[kk].a0P) + (ca[kk].a1P)*ksiD + (ca[kk].a2P)*ksiD*ksiD +(ca[kk].a3P)*pow(ksiD,3) + (ca[kk].a4P) * pow(ksiD,4);

                ak2 = (1. - (ca[kk].wo))*(3. - (ca[kk].wo)*3*(ca[kk].gc)) ;
                ak  = sqrt(ak2) ;
                e   = -3.*us*us*(ca[kk].wo) /  (4.*(1. - ak2*us*us) ) ;
                f   = -(1. - (ca[kk].wo))*3.*(ca[kk].gc)*us*us*(ca[kk].wo) / (4.*(1. - ak2*us*us) ) ;
                dp  = e / (3.*us) + us*f ;
                d   = e + f ;
                b   = 2.*ak / (3. - (ca[kk].wo)*3*(ca[kk].gc));
                del = exp( ak*taup )*(1. + b)*(1. + b) - exp(-ak*taup)*(1. - b)*(1. - b) ;
                ww  = (ca[kk].wo)/4.;
                ss  = us / (1. - ak2*us*us) ;
                q1  = 2. + 3.*us + (1. - (ca[kk].wo))*3.*(ca[kk].gc)*us*(1. + 2.*us) ;
                q2  = 2. - 3.*us - (1. - (ca[kk].wo))*3.*(ca[kk].gc)*us*(1. - 2.*us) ;
                q3  = q2*exp( -taup/us ) ;
                c1  =  ((ww*ss) / del) * ( q1*exp(ak*taup)*(1. + b) + q3*(1. - b) ) ;
                c2  = -((ww*ss) / del) * (q1*exp(-ak*taup)*(1. - b) + q3*(1. + b) ) ;
                cp1 =  c1*ak / ( 3. - (ca[kk].wo)*3.*(ca[kk].gc) ) ;
                cp2 = -c2*ak / ( 3. - (ca[kk].wo)*3.*(ca[kk].gc) ) ;
                z   = d - (ca[kk].wo)*3.*(ca[kk].gc)*uv*dp + (ca[kk].wo)*aer_phase/4. ;
                x   = c1 - (ca[kk].wo)*3.*(ca[kk].gc)*uv*cp1 ;
                y   = c2 - (ca[kk].wo)*3.*(ca[kk].gc)*uv*cp2 ;
                aa1 = uv / (1. + ak*uv) ;
                aa2 = uv / (1. - ak*uv) ;
                aa3 = us*uv / (us + uv) ;

                aer_ref = x*aa1* (1. - exp( -taup/aa1 ) ) ;
                aer_ref = aer_ref + y*aa2*( 1. - exp( -taup / aa2 )  ) ;
                aer_ref = aer_ref + z*aa3*( 1. - exp( -taup / aa3 )  ) ;
                aer_ref = aer_ref / ( us*uv );

                /*--------Residu Aerosol --------*/
                Res_aer= ( (ca[kk].Resa1) + (ca[kk].Resa2) * ( taup * m *cksi ) + (ca[kk].Resa3) * pow( (taup*m*cksi ),2) ) 
                         + (ca[kk].Resa4) * pow( (taup*m*cksi),3);


                /*---------Residu 6s-----------*/
                Res_6s= ( (ca[kk].Rest1) + (ca[kk].Rest2) * ( tautot * m *cksi )
                    + (ca[kk].Rest3) * pow( (tautot*m*cksi),2) ) + (ca[kk].Rest4) * pow( (tautot*m*cksi),3);

                /*------  11) total atmospheric reflectance */
                atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s;

                /*-------- gaseous transmission */

                tg      = th2o * to3 * to2 * tco2 * tch4* tco * tno2 ;

                /*-------- reflectance at toa*/
                /*------------------------*/
                /* Atmospheric scattering transmittance */
                //rsurf[ii] = rtoa - (atm_ref * tg) ;
                float ax1 = F1_rtls(tetas*cdr, tetav*cdr, dphi);
                float ax2 = F2_rtls(tetas*cdr, tetav*cdr, dphi);
                float f1_bar_down = ca[kk].f1d0 + ca[kk].f1d1*ax1 + ca[kk].f1d2*ax2; 
                float f2_bar_down = ca[kk].f2d0 + ca[kk].f2d1*ax1 + ca[kk].f2d2*ax2; 
                float f1_bar_bar  = ca[kk].f1b0 + ca[kk].f1b1*ax1 + ca[kk].f1b2*ax2; 
                float f2_bar_bar  = ca[kk].f2b0 + ca[kk].f2b1*ax1 + ca[kk].f2b2*ax2; 
                float rs          =  (1. + k1p*ax1 + k2p*ax2);
                ax1 = F1_rtls(tetav*cdr, tetas*cdr, M_PI-dphi);
                ax2 = F2_rtls(tetav*cdr, tetas*cdr, M_PI-dphi);
                float f1_bar_up   = ca[kk].f1d0 + ca[kk].f1d1*ax1 + ca[kk].f1d2*ax2; 
                float f2_bar_up   = ca[kk].f2d0 + ca[kk].f2d1*ax1 + ca[kk].f2d2*ax2; 
                /**/
                float ref_surf_bar_downN= (1. + k1p*f1_bar_down + k2p*f2_bar_down)/rs;
                float ref_surf_bar_upN  = (1. + k1p*f1_bar_up   + k2p*f2_bar_up)  /rs;
                float ref_surf_bar_barN = (1. + k1p*f1_bar_bar  + k2p*f2_bar_bar )/rs;
                trans_atm = (tdirtetav*tdirtetas) +
                            (tdirtetav*tdiftetas) * (ref_surf_bar_downN) +
                            (tdiftetav*tdirtetas) * (ref_surf_bar_upN) +
                            (tdiftetav*tdiftetas) * (ref_surf_bar_barN);
                //rsurf[ii] = rsurf[ii] / ( (tg * trans_atm) + (rsurf[ii] * s) ) ;
                rtoa[ii] = (rsurf * trans_atm /(1. - rsurf*s)  + atm_ref) * tg ;
  
                /* Analytical Jacobian of toa reflectance vs surface reflectance*/
                /*------------------------*/
                float ttt   = tg * trans_atm;
                //float ttt   = tg * ttetas * ttetav;
                float rps   = s * rsurf;
                float nu_dir = 1./(1. - rps);
                Jr[ii] = ttt * nu_dir*nu_dir;
       
                /* Analytical Jacobian of surface reflectance vs ozone column*/
                /*------------------------*/
                float tgp  = tg/to3;
                float dtdu = (ca[kk].ao3*ca[kk].no3/uo3) * pow ( (uo3 *m) , (ca[kk].no3) ) * to3;
                //float drdt = delta * ( (-atm_ref * tgp)  +
                //             rp * (atm_ref * s * tg - ttt)/to3 * delta);
                float drdt = rtoa[ii]/to3;
                Juo3[ii]  = drdt * dtdu;

                /* Analytical Jacobian of surface reflectance vs water vapour column*/
                /*------------------------*/
                tgp  = tg/th2o;
                dtdu = (ca[kk].ah2o*ca[kk].nh2o/uh2o) * pow ( (uh2o *m) , (ca[kk].nh2o) ) * th2o;
                //drdt = delta * ( (-atm_ref * tgp)  +
                //             rp * (atm_ref * s * tg - ttt)/th2o * delta);
                drdt = rtoa[ii]/th2o;
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


