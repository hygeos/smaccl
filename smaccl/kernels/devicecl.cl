/*
 * devicecl.cl -- SMAC atmospheric-correction OpenCL kernels.
 *
 * Kernels
 * -------
 * smaccl : inverse model. Retrieves surface (top-of-canopy) reflectance from
 *          top-of-atmosphere reflectance, together with an aerosol-model
 *          ensemble spread and the analytical / finite-difference Jacobians.
 * smaccl_dir : forward (direct) model. Propagates a surface reflectance to
 *          top-of-atmosphere reflectance with the matching Jacobians.
 *
 * Both kernels share the SMAC (6S-based) atmospheric parametrisation stored in
 * `coef_atmos` and the Ross-Thick / Li-Sparse (RTLS) BRDF kernels F1_rtls and
 * F2_rtls.
 */

#define PI 3.1415927F
#define DEUXPI 6.2831853F
#define DEMIPI 1.5707963F

/* Per-band SMAC atmospheric coefficients (6S fit) for one aerosol model; the
   host passes an array of NBAND * NMOD such structs, indexed band-major. */
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

/* ===================== Ross-Thick / Li-Sparse BRDF ===================== */

/*
 * F1_rtls -- Ross-Thick / Li-Sparse geometric (F1) BRDF kernel.
 *
 * Parameters
 * ----------
 * ths : float
 *     Source (solar) zenith angle [radians].
 * thv : float
 *     View zenith angle [radians].
 * phi : float
 *     Relative azimuth angle [radians]; folded into [0, PI] internally.
 *
 * Returns
 * -------
 * float
 *     Li-Sparse geometric kernel value F1(ths, thv, phi).
 */
float F1_rtls(float ths, float thv, float phi ){
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


/*
 * F2_rtls -- Ross-Thick / Li-Sparse volumetric (F2) BRDF kernel.
 *
 * Parameters
 * ----------
 * ths : float
 *     Source (solar) zenith angle [radians].
 * thv : float
 *     View zenith angle [radians].
 * phi : float
 *     Relative azimuth angle [radians]; folded into [0, PI] internally.
 *
 * Returns
 * -------
 * float
 *     Ross-Thick volumetric kernel value F2(ths, thv, phi).
 */
float F2_rtls(float ths, float thv, float phi ){
    if (phi < 0.) phi += DEUXPI; 
    if (phi > PI) phi = DEUXPI - phi; 
    float cos_xi = cos(ths) * cos(thv) + sin(ths) * sin(thv) * cos(phi);
    float xi = acos(cos_xi);

    // volume-scattering kernel
    float F2 = (((PI/2. -xi)*cos_xi + sin(xi))/(cos(thv) + cos(ths))) - PI/4.;

    return F2;
}

/*
 * smaccl -- SMAC inverse atmospheric correction (TOA -> surface reflectance).
 *
 * One work-item processes one image column of `NZd` pixels. For every pixel it
 * loops over the `NBANDd` spectral bands and, for each band, over an ensemble of
 * `Naerod` aerosol models; the first model (ia == 0) yields the retrieved
 * surface reflectance and its Jacobians, while the remaining models feed the
 * ensemble standard deviation. `NRUN = 3 + NBLOOPd` inner runs evaluate the
 * reference case plus the pressure / AOT(550) perturbations used for the
 * finite-difference Jacobians.
 *
 * Parameters
 * ----------
 * ca : __global coef_atmos*
 *     SMAC coefficients, NBANDd * NMOD structs (band-major).
 * tetas_, tetav_ : __global float*
 *     Solar and view zenith angles [degrees], length NZd * M.
 * phis_, phiv_ : __global float*
 *     Solar and view azimuth angles [degrees].
 * uh2o_, uo3_ : __global float*
 *     Water-vapour (g/cm2) and ozone (cm.atm) columns.
 * taup550_ : __global float*
 *     Aerosol optical thickness at 550 nm.
 * pression_ : __global float*
 *     Surface pressure [hPa].
 * rtoa_ : __global float*
 *     Top-of-atmosphere reflectance, NBANDd * NZd * M.
 * k1p_, k2p_ : __global float*
 *     RTLS BRDF ratios k1/k0, k2/k0 (0 for a Lambertian surface).
 * iaero : __global short*
 *     Aerosol-model index per pixel and ensemble member, Naerod * NZd * M.
 * NMOD, NBLOOPd, NBANDd, NZd, Naerod : int
 *     Model count per band, extra Monte-Carlo runs, band / pixel-depth / ensemble sizes.
 *
 * Returns
 * -------
 * rsurf, rsurf_0 : __global float*
 *     Surface reflectance with and without the BRDF coupling term.
 * dev_std : __global float*
 *     Aerosol-model ensemble standard deviation of the surface reflectance.
 * Jr, Juo3, Juh2o, Jpre, Jtaup : __global float*
 *     Jacobians of surface reflectance w.r.t. TOA reflectance, ozone, water
 *     vapour, pressure and AOT(550).
 */
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

    // loop over the third pixel dimension (Z)
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

        /* RTLS BRDF geometric (F1) / volumetric (F2) kernels depend ONLY on the
           pixel geometry (tetas, tetav, dphi). Hoisted out of the band/aerosol/
           run loops where they were recomputed 4 x NBAND x Naero x NRUN times
           per pixel. Bit-identical (pure function of the angles). */
        float thetas_r = tetas*cdr, thetav_r = tetav*cdr;
        float ax1d = F1_rtls(thetas_r, thetav_r, dphi);
        float ax2d = F2_rtls(thetas_r, thetav_r, dphi);
        float ax1u = F1_rtls(thetav_r, thetas_r, M_PI-dphi);
        float ax2u = F2_rtls(thetav_r, thetas_r, M_PI-dphi);

            // loop on the number of bands
        for (int ib=0; ib<XNBANDd; ib++) {
            unsigned long int ii = idx + ip*M + ib*M*XNZd;
            float rtoa=rtoa_[ii];
            float k1p = k1p_[ii];
            float k2p = k2p_[ii];

            /* Gaseous transmission is aerosol-model INVARIANT (the gas absorption
               coefficients in ca do not depend on the aerosol model), so compute
               it ONCE per band using kk0 instead of Naero times inside the model
               loop. to3/th2o are pressure-invariant; the other gases depend on
               pressure, which is perturbed only for ir==0 -> precompute the full
               gas product tg for the base and the pressure-perturbed cases. The
               multiplication order matches the original tg exactly. */
            unsigned long int kk0 = ib*NMOD;
            float to3_g = 1., th2o_g = 1., tg_base = 1., tg_pert = 1.;
            if( (uh2o> 0.) || ( uo3 > 0.) ) {
                to3_g  = exp ( (ca[kk0].ao3)  * pow ( (uo3 *m)  , (ca[kk0].no3)  ) ) ;
                th2o_g = exp ( (ca[kk0].ah2o) * pow ( (uh2o*m)  , (ca[kk0].nh2o) ) ) ;
            }
            for (int ig=0; ig<2; ig++) {
                float prg = (ig==0) ? (pression_[jj]-dpre) : pression_[jj];
                float Peqg = prg/1013.0;
                float to2g=1., tco2g=1., tch4g=1., tno2g=1., tcog=1.;
                if( (uh2o> 0.) || ( uo3 > 0.) ) {
                    float uo2g = pow (Peqg , (ca[kk0].po2));
                    float uco2g= pow (Peqg , (ca[kk0].pco2));
                    float uch4g= pow (Peqg , (ca[kk0].pch4));
                    float uno2g= pow (Peqg , (ca[kk0].pno2));
                    float ucog = pow (Peqg , (ca[kk0].pco));
                    to2g  = exp ( (ca[kk0].ao2)  * pow ( (uo2g *m)  , (ca[kk0].no2)  ) ) ;
                    tco2g = exp ( (ca[kk0].aco2) * pow ( (uco2g*m)  , (ca[kk0].nco2) ) ) ;
                    tch4g = exp ( (ca[kk0].ach4) * pow ( (uch4g*m)  , (ca[kk0].nch4) ) ) ;
                    tno2g = exp ( (ca[kk0].ano2) * pow ( (uno2g*m)  , (ca[kk0].nno2) ) ) ;
                    tcog  = exp ( (ca[kk0].aco)  * pow ( (ucog *m)  , (ca[kk0].nco)  ) ) ;
                }
                float tgg = th2o_g * to3_g * to2g * tco2g * tch4g* tcog * tno2g ;
                if (ig==0) tg_pert = tgg; else tg_base = tgg;
            }

            toc_mean = 0;
            toc_std = 0;
            for (int ia=0; ia<XNaerod; ia++) {
                iAero = iaero[jj + ia*M*XNZd];
                /* runs: reference + pressure/AOT perturbations (finite-diff Jacobians) */
                for (int ir=0; ir<NRUN; ir++) {
                    unsigned long int kk = ib*NMOD+iAero;
                    float taup550=taup550_[jj], pression=pression_[jj];

                    float dtau = dtau_rel * taup550;
                    if (ir==0) pression -= dpre;
                    if (ir==1) taup550  -= dtau;
                    Peq=pression/1013.0;
      
                    /*------  2) aerosol optical depth in the spectral band, taup  */
                    taup = (ca[kk].a0taup) + (ca[kk].a1taup) * taup550 ;

                    /*------  3,4) gaseous transmissions: precomputed once per band
                       (aerosol-model invariant); to3/th2o reused for the ozone /
                       water-vapour Jacobians below. */
                    to3 = to3_g; th2o = th2o_g;

                    /*------  5) Total scattering transmission */
                    ttetas = (ca[kk].a0T) + (ca[kk].a1T)*taup550/us + ((ca[kk].a2T)*Peq + (ca[kk].a3T))/(1.+us) ; /* downward */
                    ttetav = (ca[kk].a0T) + (ca[kk].a1T)*taup550/uv + ((ca[kk].a2T)*Peq + (ca[kk].a3T))/(1.+uv) ; /* upward   */

                    /*------  6) spherical albedo of the atmosphere */
                    s = (ca[kk].a0s) * Peq +  (ca[kk].a3s) + (ca[kk].a1s)*taup550 + (ca[kk].a2s) *taup550*taup550 ;

                    taurz=(ca[kk].taur)*Peq;

                    ray_ref   = ( taurz*ray_phase ) / (4.*us*uv) ;

                    /*-----------------Residu Rayleigh ---------*/
                    float rr_ray = taurz*ray_phase/(us*uv);
                    Res_ray= (ca[kk].Resr1) + (ca[kk].Resr2) * rr_ray +
                     (ca[kk].Resr3) * rr_ray*rr_ray;

                    /*--------9b) Direct and scattering transmssions*/
                    tautot=taup+taurz;
                    tdirtetas = exp(-tautot/us);
                    tdirtetav = exp(-tautot/uv);
                    tdiftetas = ttetas - tdirtetas;
                    tdiftetav = ttetav - tdirtetav;

                    /*------  10) aerosol atmospheric reflectance */
                    float ksi2 = ksiD*ksiD;
                    aer_phase = (ca[kk].a0P) + (ca[kk].a1P)*ksiD + (ca[kk].a2P)*ksi2 +(ca[kk].a3P)*ksi2*ksiD + (ca[kk].a4P) * ksi2*ksi2;

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
                    float tmc = taup*m*cksi;
                    Res_aer= ( (ca[kk].Resa1) + (ca[kk].Resa2) * tmc + (ca[kk].Resa3) * tmc*tmc ) 
                             + (ca[kk].Resa4) * tmc*tmc*tmc;


                    /*---------Residu 6s-----------*/
                    float ttmc = tautot*m*cksi;
                    Res_6s= ( (ca[kk].Rest1) + (ca[kk].Rest2) * ttmc
                        + (ca[kk].Rest3) * ttmc*ttmc ) + (ca[kk].Rest4) * ttmc*ttmc*ttmc;

                    /*------  11) total atmospheric reflectance */
                    atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s;

                    /*-------- reflectance at toa*/

                    tg      = (ir==0) ? tg_pert : tg_base ;

                     /* reflectance at surface */
                    /*------------------------*/
                    toc = rtoa - (atm_ref * tg) ;
                    float f1_bar_down = ca[kk].f1d0 + ca[kk].f1d1*ax1d + ca[kk].f1d2*ax2d; 
                    float f2_bar_down = ca[kk].f2d0 + ca[kk].f2d1*ax1d + ca[kk].f2d2*ax2d; 
                    float f1_bar_bar  = ca[kk].f1b0 + ca[kk].f1b1*ax1d + ca[kk].f1b2*ax2d; 
                    float f2_bar_bar  = ca[kk].f2b0 + ca[kk].f2b1*ax1d + ca[kk].f2b2*ax2d; 
                    float rs          =  (1. + k1p*ax1d + k2p*ax2d);
                    float f1_bar_up   = ca[kk].f1d0 + ca[kk].f1d1*ax1u + ca[kk].f1d2*ax2u; 
                    float f2_bar_up   = ca[kk].f2d0 + ca[kk].f2d1*ax1u + ca[kk].f2d2*ax2u; 
                    float ref_surf_bar_downN= (1. + k1p*f1_bar_down + k2p*f2_bar_down)/rs;
                    float ref_surf_bar_upN  = (1. + k1p*f1_bar_up   + k2p*f2_bar_up)  /rs;
                    float ref_surf_bar_barN = (1. + k1p*f1_bar_bar  + k2p*f2_bar_bar )/rs;
                    trans_atm = (tdirtetav*tdirtetas) +
                                (tdirtetav*tdiftetas) * (ref_surf_bar_downN) +
                                (tdiftetav*tdirtetas) * (ref_surf_bar_upN) +
                                (tdiftetav*tdiftetas) * (ref_surf_bar_barN);
                    trans_atm_0 = (tdirtetav*tdirtetas) + (tdirtetav*tdiftetas) + (tdiftetav*tdirtetas) + (tdiftetav*tdiftetas);
                    toc_0 = toc / ( (tg * trans_atm_0) + (toc * s) ) ;
                    toc = toc / ( (tg * trans_atm) + (toc * s) ) ;
                    
                    if (ia==0) {
                        rsurf[ii] = toc;
                        rsurf_0[ii] = toc_0;
                        /* Analytical Jacobian of surface reflectance vs toa reflectance*/
                        /*------------------------*/
                        float ttt   = tg * trans_atm;
                        float rp    = rtoa - atm_ref * tg;
                        float rps   = s * rp;
                        float nu_inv = 1./(rps + ttt);
                        Jr[ii] = nu_inv *  nu_inv * ttt ;
               
                        /* Analytical Jacobian of surface reflectance vs ozone column*/
                        /*------------------------*/
                        float tgp  = tg/to3;
                        float dtdu = (ca[kk].ao3*ca[kk].no3/uo3) * pow ( (uo3 *m) , (ca[kk].no3) ) * to3;
                        float drdt =  -rtoa * nu_inv*nu_inv * ttt/to3;
                        Juo3[ii]  = drdt * dtdu;

                        /* Analytical Jacobian of surface reflectance vs water vapour column*/
                        /*------------------------*/
                        tgp  = tg/th2o;
                        dtdu = (ca[kk].ah2o*ca[kk].nh2o/uh2o) * pow ( (uh2o *m) , (ca[kk].nh2o) ) * th2o;
                        drdt =  -rtoa * nu_inv*nu_inv * ttt/th2o;
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
                    }
                } // main loop (ir)
                if (ia!=0) {
                    toc_mean += toc;
                    toc_std  += toc*toc;
                }
            } // main loop (ia)
            toc_mean /= (float)(XNaerod -1);
            toc_std /= (float)(XNaerod -1);
            dev_std[ii] = sqrt( fabs(toc_std - toc_mean*toc_mean ));
        } // main loop (ib)
    } // main loop (ip)

}


/*
 * smaccl_dir -- SMAC forward (direct) model (surface -> TOA reflectance).
 *
 * The mirror of `smaccl`: one work-item processes one image column of `NZd`
 * pixels and, for every pixel, loops over `NRUN = 3 + NBLOOPd` runs (reference
 * plus pressure / AOT(550) perturbations for the finite-difference Jacobians)
 * and the `NBANDd` spectral bands. A single aerosol model per pixel is used
 * (no ensemble). Pixel geometry and the RTLS BRDF kernels are computed once per
 * pixel, as in `smaccl`.
 *
 * Parameters
 * ----------
 * ca : __global coef_atmos*
 *     SMAC coefficients, NBANDd * NMOD structs (band-major).
 * tetas_, tetav_, phis_, phiv_ : __global float*
 *     Solar / view zenith and azimuth angles [degrees].
 * uh2o_, uo3_ : __global float*
 *     Water-vapour (g/cm2) and ozone (cm.atm) columns.
 * taup550_ : __global float*
 *     Aerosol optical thickness at 550 nm.
 * pression_ : __global float*
 *     Surface pressure [hPa].
 * rsurf_ : __global float*
 *     Input surface (top-of-canopy) reflectance, NBANDd * NZd * M.
 * k1p_, k2p_ : __global float*
 *     RTLS BRDF ratios k1/k0, k2/k0 (0 for a Lambertian surface).
 * iaero : __global int*
 *     Aerosol-model index per pixel.
 * NMOD, NBLOOPd, NBANDd, NZd : int
 *     Model count per band, extra Monte-Carlo runs, band and pixel-depth sizes.
 *
 * Returns
 * -------
 * rtoa : __global float*
 *     Modelled top-of-atmosphere reflectance.
 * Jr, Juo3, Juh2o, Jpre, Jtaup : __global float*
 *     Jacobians of TOA reflectance w.r.t. surface reflectance, ozone, water
 *     vapour, pressure and AOT(550).
 */
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

    const int XXBLOCKd=gs0;
    const int XXGRIDd=gs1;
    const int XNBLOOPd=NBLOOPd;
    const int XNZd=NZd;
    const int XNBANDd=NBANDd;

    const int idx = gid0 * gs1 + gid1;
    const int NRUN=3 + XNBLOOPd; // reference run + pressure/AOT perturbations (+ MC loops)
    const int M=XXBLOCKd * XXGRIDd;
    const float dpre = 10.;      // absolute pressure perturbation (hPa) for Jacobian
    const float dtau_rel = 0.1;  // relative AOT(550) perturbation for Jacobian

    const float crd=180./M_PI;
    const float cdr=M_PI/180.;

    /* SMAC scratch variables */
    float cksi, s, m, tg, us, uv, dphi;
    float to3, th2o, to2, tco2, tco, tno2, tch4;
    float ttetas, ttetav, ksiD;
    float tdirtetas, tdirtetav, tdiftetas, tdiftetav, trans_atm;
    float atm_ref;
    float ak2, ak, e, f, dp, d, b, del, ww, ss, q1, q2, q3, c1, c2, cp1;
    float cp2, z, x, y, aa1, aa2, aa3;
    float taup, tautot, taurz;
    float Peq;
    float Res_ray, Res_aer, Res_6s;
    float ray_phase, ray_ref, aer_ref, aer_phase;
    int iAero = 0;

    // loop over the third pixel dimension (Z)
    for (int ip=0; ip<XNZd; ip++) {
        unsigned long int jj = idx + ip*M;
        iAero = iaero[jj];
        float tetas=tetas_[jj], tetav=tetav_[jj], phis=phis_[jj], phiv=phiv_[jj], uh2o=uh2o_[jj], uo3=uo3_[jj];

        /* pixel geometry (independent of band, aerosol model and run) */
        us = cos(tetas*cdr);
        uv = cos(tetav*cdr);
        dphi = (phis-phiv)*cdr;
        m = 1./us + 1./uv;
        cksi = - ( (us*uv) + (sqrt(1. - us*us) * sqrt(1. - uv*uv) * cos(dphi)) );
        if (cksi < -1) cksi = -1.0;
        ksiD = crd*acos(cksi);
        ray_phase = 0.7190443 * (1. + cksi*cksi) + 0.0412742;

        /* RTLS BRDF kernels depend only on the geometry -- hoisted per pixel */
        float ax1d = F1_rtls(tetas*cdr, tetav*cdr, dphi);
        float ax2d = F2_rtls(tetas*cdr, tetav*cdr, dphi);
        float ax1u = F1_rtls(tetav*cdr, tetas*cdr, M_PI-dphi);
        float ax2u = F2_rtls(tetav*cdr, tetas*cdr, M_PI-dphi);

        /* runs: reference + pressure/AOT perturbations (finite-diff Jacobians) */
        for (int ir=0; ir<NRUN; ir++) {
            for (int ib=0; ib<XNBANDd; ib++) {
                unsigned long int ii = idx + ip*M + ib*M*XNZd;
                unsigned long int kk = ib*NMOD+iAero;
                float taup550=taup550_[jj], pression=pression_[jj];
                float rsurf=rsurf_[ii];
                float k1p = k1p_[ii];
                float k2p = k2p_[ii];

                float dtau = dtau_rel * taup550;
                if (ir==0) pression -= dpre;
                if (ir==1) taup550  -= dtau;
                Peq = pression/1013.0;

                /*------ 2) aerosol optical depth in the spectral band, taup */
                taup = (ca[kk].a0taup) + (ca[kk].a1taup) * taup550 ;

                /*------ 3,4) gaseous transmissions (no absorption if uh2o<=0 and uo3<=0) */
                to3 = 1.; th2o = 1.; to2 = 1.; tco2 = 1.; tch4 = 1.; tno2 = 1.; tco = 1.;
                if( (uh2o > 0.) || (uo3 > 0.) ) {
                    float uo2  = pow(Peq, (ca[kk].po2));
                    float uco2 = pow(Peq, (ca[kk].pco2));
                    float uch4 = pow(Peq, (ca[kk].pch4));
                    float uno2 = pow(Peq, (ca[kk].pno2));
                    float uco  = pow(Peq, (ca[kk].pco));
                    to3  = exp((ca[kk].ao3)  * pow((uo3 *m), (ca[kk].no3)));
                    th2o = exp((ca[kk].ah2o) * pow((uh2o*m), (ca[kk].nh2o)));
                    to2  = exp((ca[kk].ao2)  * pow((uo2 *m), (ca[kk].no2)));
                    tco2 = exp((ca[kk].aco2) * pow((uco2*m), (ca[kk].nco2)));
                    tch4 = exp((ca[kk].ach4) * pow((uch4*m), (ca[kk].nch4)));
                    tno2 = exp((ca[kk].ano2) * pow((uno2*m), (ca[kk].nno2)));
                    tco  = exp((ca[kk].aco)  * pow((uco *m), (ca[kk].nco)));
                }

                /*------ 5) total scattering transmission */
                ttetas = (ca[kk].a0T) + (ca[kk].a1T)*taup550/us + ((ca[kk].a2T)*Peq + (ca[kk].a3T))/(1.+us) ;
                ttetav = (ca[kk].a0T) + (ca[kk].a1T)*taup550/uv + ((ca[kk].a2T)*Peq + (ca[kk].a3T))/(1.+uv) ;

                /*------ 6) spherical albedo of the atmosphere */
                s = (ca[kk].a0s) * Peq + (ca[kk].a3s) + (ca[kk].a1s)*taup550 + (ca[kk].a2s)*taup550*taup550 ;

                /*------ 9) Rayleigh atmospheric reflectance and residual */
                taurz = (ca[kk].taur)*Peq;
                ray_ref = (taurz*ray_phase) / (4.*us*uv);
                float rr_ray = taurz*ray_phase/(us*uv);
                Res_ray = (ca[kk].Resr1) + (ca[kk].Resr2)*rr_ray + (ca[kk].Resr3)*rr_ray*rr_ray;

                /*------ 9b) direct and diffuse transmissions */
                tautot = taup+taurz;
                tdirtetas = exp(-tautot/us);
                tdirtetav = exp(-tautot/uv);
                tdiftetas = ttetas - tdirtetas;
                tdiftetav = ttetav - tdirtetav;

                /*------ 10) aerosol atmospheric reflectance */
                float ksi2 = ksiD*ksiD;
                aer_phase = (ca[kk].a0P) + (ca[kk].a1P)*ksiD + (ca[kk].a2P)*ksi2 + (ca[kk].a3P)*ksi2*ksiD + (ca[kk].a4P)*ksi2*ksi2;

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

                /*------ aerosol and 6S residuals */
                float tmc = taup*m*cksi;
                Res_aer = ((ca[kk].Resa1) + (ca[kk].Resa2)*tmc + (ca[kk].Resa3)*tmc*tmc) + (ca[kk].Resa4)*tmc*tmc*tmc;
                float ttmc = tautot*m*cksi;
                Res_6s  = ((ca[kk].Rest1) + (ca[kk].Rest2)*ttmc + (ca[kk].Rest3)*ttmc*ttmc) + (ca[kk].Rest4)*ttmc*ttmc*ttmc;

                /*------ 11) total atmospheric reflectance and gaseous transmission */
                atm_ref = ray_ref - Res_ray + aer_ref - Res_aer + Res_6s;
                tg = th2o * to3 * to2 * tco2 * tch4 * tco * tno2 ;

                /*------ surface BRDF coupling (RTLS kernels hoisted per pixel) */
                float f1_bar_down = ca[kk].f1d0 + ca[kk].f1d1*ax1d + ca[kk].f1d2*ax2d;
                float f2_bar_down = ca[kk].f2d0 + ca[kk].f2d1*ax1d + ca[kk].f2d2*ax2d;
                float f1_bar_bar  = ca[kk].f1b0 + ca[kk].f1b1*ax1d + ca[kk].f1b2*ax2d;
                float f2_bar_bar  = ca[kk].f2b0 + ca[kk].f2b1*ax1d + ca[kk].f2b2*ax2d;
                float rs          = (1. + k1p*ax1d + k2p*ax2d);
                float f1_bar_up   = ca[kk].f1d0 + ca[kk].f1d1*ax1u + ca[kk].f1d2*ax2u;
                float f2_bar_up   = ca[kk].f2d0 + ca[kk].f2d1*ax1u + ca[kk].f2d2*ax2u;
                float ref_surf_bar_downN = (1. + k1p*f1_bar_down + k2p*f2_bar_down)/rs;
                float ref_surf_bar_upN   = (1. + k1p*f1_bar_up   + k2p*f2_bar_up)  /rs;
                float ref_surf_bar_barN  = (1. + k1p*f1_bar_bar  + k2p*f2_bar_bar )/rs;
                trans_atm = (tdirtetav*tdirtetas) +
                            (tdirtetav*tdiftetas) * ref_surf_bar_downN +
                            (tdiftetav*tdirtetas) * ref_surf_bar_upN +
                            (tdiftetav*tdiftetas) * ref_surf_bar_barN;

                /*------ forward model: surface -> TOA reflectance */
                rtoa[ii] = (rsurf * trans_atm / (1. - rsurf*s) + atm_ref) * tg ;

                /* Analytical Jacobian of TOA reflectance vs surface reflectance */
                float ttt = tg * trans_atm;
                float rps = s * rsurf;
                float nu_dir = 1./(1. - rps);
                Jr[ii] = ttt * nu_dir*nu_dir;

                /* Analytical Jacobian vs ozone column */
                float dtdu = (ca[kk].ao3*ca[kk].no3/uo3) * pow((uo3 *m), (ca[kk].no3)) * to3;
                float drdt = rtoa[ii]/to3;
                Juo3[ii] = drdt * dtdu;

                /* Analytical Jacobian vs water-vapour column */
                dtdu = (ca[kk].ah2o*ca[kk].nh2o/uh2o) * pow((uh2o *m), (ca[kk].nh2o)) * th2o;
                drdt = rtoa[ii]/th2o;
                Juh2o[ii] = drdt * dtdu;

                /* Finite-difference Jacobians vs pressure and AOT(550) */
                if (ir==0) Jpre[ii]  = -rtoa[ii];
                if (ir==1) Jtaup[ii] = -rtoa[ii];
                if (ir==2) {
                    Jpre[ii]  += rtoa[ii];
                    Jpre[ii]  /= dpre;
                    Jtaup[ii] += rtoa[ii];
                    Jtaup[ii] /= dtau;
                }
            } // band loop (ib)
        } // run loop (ir)
    } // pixel loop (ip)
}


