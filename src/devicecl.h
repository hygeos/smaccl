#ifndef DEVICEOCL_H
#define DEVICEOCL_H

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

  } coef_atmos ;

#endif // DEVICEOCL_H

