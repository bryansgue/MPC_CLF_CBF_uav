/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Drone_ode_cost_ext_cost_0_fun_jac_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_c0 CASADI_PREFIX(c0)
#define casadi_c1 CASADI_PREFIX(c1)
#define casadi_c2 CASADI_PREFIX(c2)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_densify CASADI_PREFIX(densify)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_sq CASADI_PREFIX(sq)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

#define CASADI_CAST(x,y) ((x) y)

void casadi_densify(const casadi_real* x, const casadi_int* sp_x, casadi_real* y, casadi_int tr) {
  casadi_int nrow_x, ncol_x, i, el;
  const casadi_int *colind_x, *row_x;
  if (!y) return;
  nrow_x = sp_x[0]; ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x+ncol_x+3;
  casadi_clear(y, nrow_x*ncol_x);
  if (!x) return;
  if (tr) {
    for (i=0; i<ncol_x; ++i) {
      for (el=colind_x[i]; el!=colind_x[i+1]; ++el) {
        y[i + row_x[el]*ncol_x] = CASADI_CAST(casadi_real, *x++);
      }
    }
  } else {
    for (i=0; i<ncol_x; ++i) {
      for (el=colind_x[i]; el!=colind_x[i+1]; ++el) {
        y[row_x[el]] = CASADI_CAST(casadi_real, *x++);
      }
      y += nrow_x;
    }
  }
}

static const casadi_int casadi_s0[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
static const casadi_int casadi_s1[18] = {15, 1, 0, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
static const casadi_int casadi_s2[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s3[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s4[4] = {0, 1, 0, 0};
static const casadi_int casadi_s5[29] = {25, 1, 0, 25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
static const casadi_int casadi_s6[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s7[19] = {15, 1, 0, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};

static const casadi_real casadi_c0[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
static const casadi_real casadi_c1[9] = {2., 0., 0., 0., 2., 0., 0., 0., 2.};
static const casadi_real casadi_c2[16] = {2.5000000000000000e+00, 0., 0., 0., 0., 2.5000000000000000e+00, 0., 0., 0., 0., 5., 0., 0., 0., 0., 2.5000000000000000e+00};

/* Drone_ode_cost_ext_cost_0_fun_jac:(i0[11],i1[4],i2[0],i3[25])->(o0,o1[15]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_int *cii;
  const casadi_real *cr, *cs;
  casadi_real w0, *w1=w+4, *w2=w+7, *w3=w+10, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29, *w30=w+45, *w31=w+70, *w32=w+73, *w33=w+76, *w34=w+85, *w35=w+88, *w36=w+91, *w37=w+100, *w38=w+104, *w39=w+108, *w40=w+111, *w41=w+114, *w42=w+117, *w43=w+120, *w44=w+129, *w45=w+133, *w46=w+149, *w47=w+158, *w48=w+167, *w49=w+176, *w50=w+180, *w51=w+189, *w52=w+192, *w53=w+196, *w55=w+212, *w56=w+226;
  /* #0: @0 = 0 */
  w0 = 0.;
  /* #1: @1 = zeros(1x3) */
  casadi_clear(w1, 3);
  /* #2: @2 = zeros(3x1) */
  casadi_clear(w2, 3);
  /* #3: @3 = 
  [[1, 0, 0], 
   [0, 1, 0], 
   [0, 0, 1]] */
  casadi_copy(casadi_c0, 9, w3);
  /* #4: @4 = 0 */
  w4 = 0.;
  /* #5: @5 = input[3][0] */
  w5 = arg[3] ? arg[3][0] : 0;
  /* #6: @6 = input[3][1] */
  w6 = arg[3] ? arg[3][1] : 0;
  /* #7: @7 = input[3][2] */
  w7 = arg[3] ? arg[3][2] : 0;
  /* #8: @8 = input[3][3] */
  w8 = arg[3] ? arg[3][3] : 0;
  /* #9: @9 = input[3][4] */
  w9 = arg[3] ? arg[3][4] : 0;
  /* #10: @10 = input[3][5] */
  w10 = arg[3] ? arg[3][5] : 0;
  /* #11: @11 = input[3][6] */
  w11 = arg[3] ? arg[3][6] : 0;
  /* #12: @12 = input[3][7] */
  w12 = arg[3] ? arg[3][7] : 0;
  /* #13: @13 = input[3][8] */
  w13 = arg[3] ? arg[3][8] : 0;
  /* #14: @14 = input[3][9] */
  w14 = arg[3] ? arg[3][9] : 0;
  /* #15: @15 = input[3][10] */
  w15 = arg[3] ? arg[3][10] : 0;
  /* #16: @16 = input[3][11] */
  w16 = arg[3] ? arg[3][11] : 0;
  /* #17: @17 = input[3][12] */
  w17 = arg[3] ? arg[3][12] : 0;
  /* #18: @18 = input[3][13] */
  w18 = arg[3] ? arg[3][13] : 0;
  /* #19: @19 = input[3][14] */
  w19 = arg[3] ? arg[3][14] : 0;
  /* #20: @20 = input[3][15] */
  w20 = arg[3] ? arg[3][15] : 0;
  /* #21: @21 = input[3][16] */
  w21 = arg[3] ? arg[3][16] : 0;
  /* #22: @22 = input[3][17] */
  w22 = arg[3] ? arg[3][17] : 0;
  /* #23: @23 = input[3][18] */
  w23 = arg[3] ? arg[3][18] : 0;
  /* #24: @24 = input[3][19] */
  w24 = arg[3] ? arg[3][19] : 0;
  /* #25: @25 = input[3][20] */
  w25 = arg[3] ? arg[3][20] : 0;
  /* #26: @26 = input[3][21] */
  w26 = arg[3] ? arg[3][21] : 0;
  /* #27: @27 = input[3][22] */
  w27 = arg[3] ? arg[3][22] : 0;
  /* #28: @28 = input[3][23] */
  w28 = arg[3] ? arg[3][23] : 0;
  /* #29: @29 = input[3][24] */
  w29 = arg[3] ? arg[3][24] : 0;
  /* #30: @30 = vertcat(@5, @6, @7, @8, @9, @10, @11, @12, @13, @14, @15, @16, @17, @18, @19, @20, @21, @22, @23, @24, @25, @26, @27, @28, @29) */
  rr=w30;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  *rr++ = w12;
  *rr++ = w13;
  *rr++ = w14;
  *rr++ = w15;
  *rr++ = w16;
  *rr++ = w17;
  *rr++ = w18;
  *rr++ = w19;
  *rr++ = w20;
  *rr++ = w21;
  *rr++ = w22;
  *rr++ = w23;
  *rr++ = w24;
  *rr++ = w25;
  *rr++ = w26;
  *rr++ = w27;
  *rr++ = w28;
  *rr++ = w29;
  /* #31: @31 = @30[15:18] */
  for (rr=w31, ss=w30+15; ss!=w30+18; ss+=1) *rr++ = *ss;
  /* #32: @32 = @31' */
  casadi_copy(w31, 3, w32);
  /* #33: @4 = mac(@32,@31,@4) */
  for (i=0, rr=(&w4); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w32+j, tt=w31+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #34: @33 = @4[0, 0, 0, 0, 0, 0, 0, 0, 0] */
  for (cii=casadi_s0, rr=w33, ss=(&w4); cii!=casadi_s0+9; ++cii) *rr++ = *cii>=0 ? ss[*cii] : 0;
  /* #35: @3 = (@3-@33) */
  for (i=0, rr=w3, cs=w33; i<9; ++i) (*rr++) -= (*cs++);
  /* #36: @32 = @30[:3] */
  for (rr=w32, ss=w30+0; ss!=w30+3; ss+=1) *rr++ = *ss;
  /* #37: @4 = input[0][0] */
  w4 = arg[0] ? arg[0][0] : 0;
  /* #38: @5 = input[0][1] */
  w5 = arg[0] ? arg[0][1] : 0;
  /* #39: @6 = input[0][2] */
  w6 = arg[0] ? arg[0][2] : 0;
  /* #40: @34 = vertcat(@4, @5, @6) */
  rr=w34;
  *rr++ = w4;
  *rr++ = w5;
  *rr++ = w6;
  /* #41: @32 = (@32-@34) */
  for (i=0, rr=w32, cs=w34; i<3; ++i) (*rr++) -= (*cs++);
  /* #42: @2 = mac(@3,@32,@2) */
  for (i=0, rr=w2; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w3+j, tt=w32+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #43: @34 = @2' */
  casadi_copy(w2, 3, w34);
  /* #44: @33 = 
  [[2, 0, 0], 
   [0, 2, 0], 
   [0, 0, 2]] */
  casadi_copy(casadi_c1, 9, w33);
  /* #45: @1 = mac(@34,@33,@1) */
  for (i=0, rr=w1; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w34+j, tt=w33+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #46: @0 = mac(@1,@2,@0) */
  for (i=0, rr=(&w0); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w1+j, tt=w2+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #47: @4 = 0 */
  w4 = 0.;
  /* #48: @34 = zeros(1x3) */
  casadi_clear(w34, 3);
  /* #49: @5 = dot(@31, @32) */
  w5 = casadi_dot(3, w31, w32);
  /* #50: @32 = (@5*@31) */
  for (i=0, rr=w32, cs=w31; i<3; ++i) (*rr++)  = (w5*(*cs++));
  /* #51: @35 = @32' */
  casadi_copy(w32, 3, w35);
  /* #52: @36 = 
  [[2, 0, 0], 
   [0, 2, 0], 
   [0, 0, 2]] */
  casadi_copy(casadi_c1, 9, w36);
  /* #53: @34 = mac(@35,@36,@34) */
  for (i=0, rr=w34; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w35+j, tt=w36+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #54: @4 = mac(@34,@32,@4) */
  for (i=0, rr=(&w4); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w34+j, tt=w32+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #55: @0 = (@0+@4) */
  w0 += w4;
  /* #56: @4 = 0 */
  w4 = 0.;
  /* #57: @35 = zeros(1x3) */
  casadi_clear(w35, 3);
  /* #58: @9 = (-@9) */
  w9 = (- w9 );
  /* #59: @10 = (-@10) */
  w10 = (- w10 );
  /* #60: @11 = (-@11) */
  w11 = (- w11 );
  /* #61: @37 = vertcat(@8, @9, @10, @11) */
  rr=w37;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w10;
  *rr++ = w11;
  /* #62: @38 = @30[3:7] */
  for (rr=w38, ss=w30+3; ss!=w30+7; ss+=1) *rr++ = *ss;
  /* #63: @8 = ||@38||_F */
  w8 = sqrt(casadi_dot(4, w38, w38));
  /* #64: @37 = (@37/@8) */
  for (i=0, rr=w37; i<4; ++i) (*rr++) /= w8;
  /* #65: @8 = @37[0] */
  for (rr=(&w8), ss=w37+0; ss!=w37+1; ss+=1) *rr++ = *ss;
  /* #66: @9 = input[0][3] */
  w9 = arg[0] ? arg[0][3] : 0;
  /* #67: @10 = (@8*@9) */
  w10  = (w8*w9);
  /* #68: @11 = @37[1] */
  for (rr=(&w11), ss=w37+1; ss!=w37+2; ss+=1) *rr++ = *ss;
  /* #69: @5 = input[0][4] */
  w5 = arg[0] ? arg[0][4] : 0;
  /* #70: @6 = (@11*@5) */
  w6  = (w11*w5);
  /* #71: @10 = (@10-@6) */
  w10 -= w6;
  /* #72: @6 = @37[2] */
  for (rr=(&w6), ss=w37+2; ss!=w37+3; ss+=1) *rr++ = *ss;
  /* #73: @7 = input[0][5] */
  w7 = arg[0] ? arg[0][5] : 0;
  /* #74: @12 = (@6*@7) */
  w12  = (w6*w7);
  /* #75: @10 = (@10-@12) */
  w10 -= w12;
  /* #76: @12 = @37[3] */
  for (rr=(&w12), ss=w37+3; ss!=w37+4; ss+=1) *rr++ = *ss;
  /* #77: @13 = input[0][6] */
  w13 = arg[0] ? arg[0][6] : 0;
  /* #78: @14 = (@12*@13) */
  w14  = (w12*w13);
  /* #79: @10 = (@10-@14) */
  w10 -= w14;
  /* #80: @14 = 0 */
  w14 = 0.;
  /* #81: @14 = (@10<@14) */
  w14  = (w10<w14);
  /* #82: @15 = (@8*@5) */
  w15  = (w8*w5);
  /* #83: @16 = (@11*@9) */
  w16  = (w11*w9);
  /* #84: @15 = (@15+@16) */
  w15 += w16;
  /* #85: @16 = (@6*@13) */
  w16  = (w6*w13);
  /* #86: @15 = (@15+@16) */
  w15 += w16;
  /* #87: @16 = (@12*@7) */
  w16  = (w12*w7);
  /* #88: @15 = (@15-@16) */
  w15 -= w16;
  /* #89: @16 = (@8*@7) */
  w16  = (w8*w7);
  /* #90: @17 = (@11*@13) */
  w17  = (w11*w13);
  /* #91: @16 = (@16-@17) */
  w16 -= w17;
  /* #92: @17 = (@6*@9) */
  w17  = (w6*w9);
  /* #93: @16 = (@16+@17) */
  w16 += w17;
  /* #94: @17 = (@12*@5) */
  w17  = (w12*w5);
  /* #95: @16 = (@16+@17) */
  w16 += w17;
  /* #96: @17 = (@8*@13) */
  w17  = (w8*w13);
  /* #97: @18 = (@11*@7) */
  w18  = (w11*w7);
  /* #98: @17 = (@17+@18) */
  w17 += w18;
  /* #99: @18 = (@6*@5) */
  w18  = (w6*w5);
  /* #100: @17 = (@17-@18) */
  w17 -= w18;
  /* #101: @18 = (@12*@9) */
  w18  = (w12*w9);
  /* #102: @17 = (@17+@18) */
  w17 += w18;
  /* #103: @37 = vertcat(@10, @15, @16, @17) */
  rr=w37;
  *rr++ = w10;
  *rr++ = w15;
  *rr++ = w16;
  *rr++ = w17;
  /* #104: @38 = (-@37) */
  for (i=0, rr=w38, cs=w37; i<4; ++i) *rr++ = (- *cs++ );
  /* #105: @38 = (@14?@38:0) */
  for (i=0, rr=w38, cs=w38; i<4; ++i) (*rr++)  = (w14?(*cs++):0);
  /* #106: @10 = (!@14) */
  w10 = (! w14 );
  /* #107: @37 = (@10?@37:0) */
  for (i=0, rr=w37, cs=w37; i<4; ++i) (*rr++)  = (w10?(*cs++):0);
  /* #108: @38 = (@38+@37) */
  for (i=0, rr=w38, cs=w37; i<4; ++i) (*rr++) += (*cs++);
  /* #109: @39 = @38[1:4] */
  for (rr=w39, ss=w38+1; ss!=w38+4; ss+=1) *rr++ = *ss;
  /* #110: @40 = (2.*@39) */
  for (i=0, rr=w40, cs=w39; i<3; ++i) *rr++ = (2.* *cs++ );
  /* #111: @15 = ||@39||_F */
  w15 = sqrt(casadi_dot(3, w39, w39));
  /* #112: @16 = @38[0] */
  for (rr=(&w16), ss=w38+0; ss!=w38+1; ss+=1) *rr++ = *ss;
  /* #113: @17 = atan2(@15,@16) */
  w17  = atan2(w15,w16);
  /* #114: @41 = (@40*@17) */
  for (i=0, rr=w41, cr=w40; i<3; ++i) (*rr++)  = ((*cr++)*w17);
  /* #115: @41 = (@41/@15) */
  for (i=0, rr=w41; i<3; ++i) (*rr++) /= w15;
  /* #116: @42 = @41' */
  casadi_copy(w41, 3, w42);
  /* #117: @43 = 
  [[1, 0, 0], 
   [0, 1, 0], 
   [0, 0, 1]] */
  casadi_copy(casadi_c0, 9, w43);
  /* #118: @35 = mac(@42,@43,@35) */
  for (i=0, rr=w35; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w42+j, tt=w43+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #119: @4 = mac(@35,@41,@4) */
  for (i=0, rr=(&w4); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w35+j, tt=w41+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #120: @0 = (@0+@4) */
  w0 += w4;
  /* #121: @4 = 0 */
  w4 = 0.;
  /* #122: @38 = zeros(1x4) */
  casadi_clear(w38, 4);
  /* #123: @18 = input[1][0] */
  w18 = arg[1] ? arg[1][0] : 0;
  /* #124: @19 = input[1][1] */
  w19 = arg[1] ? arg[1][1] : 0;
  /* #125: @20 = input[1][2] */
  w20 = arg[1] ? arg[1][2] : 0;
  /* #126: @21 = input[1][3] */
  w21 = arg[1] ? arg[1][3] : 0;
  /* #127: @37 = vertcat(@18, @19, @20, @21) */
  rr=w37;
  *rr++ = w18;
  *rr++ = w19;
  *rr++ = w20;
  *rr++ = w21;
  /* #128: @44 = @37' */
  casadi_copy(w37, 4, w44);
  /* #129: @45 = 
  [[2.5, 0, 0, 0], 
   [0, 2.5, 0, 0], 
   [0, 0, 5, 0], 
   [0, 0, 0, 2.5]] */
  casadi_copy(casadi_c2, 16, w45);
  /* #130: @38 = mac(@44,@45,@38) */
  for (i=0, rr=w38; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w44+j, tt=w45+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #131: @4 = mac(@38,@37,@4) */
  for (i=0, rr=(&w4); i<1; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w38+j, tt=w37+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #132: @0 = (@0+@4) */
  w0 += w4;
  /* #133: @4 = 0.0001 */
  w4 = 1.0000000000000000e-04;
  /* #134: @42 = zeros(3x1) */
  casadi_clear(w42, 3);
  /* #135: @46 = 
  [[1, 0, 0], 
   [0, 1, 0], 
   [0, 0, 1]] */
  casadi_copy(casadi_c0, 9, w46);
  /* #136: @47 = zeros(3x3) */
  casadi_clear(w47, 9);
  /* #137: @48 = zeros(3x3) */
  casadi_clear(w48, 9);
  /* #138: @44 = vertcat(@9, @5, @7, @13) */
  rr=w44;
  *rr++ = w9;
  *rr++ = w5;
  *rr++ = w7;
  *rr++ = w13;
  /* #139: @9 = ||@44||_F */
  w9 = sqrt(casadi_dot(4, w44, w44));
  /* #140: @49 = (@44/@9) */
  for (i=0, rr=w49, cr=w44; i<4; ++i) (*rr++)  = ((*cr++)/w9);
  /* #141: @5 = @49[3] */
  for (rr=(&w5), ss=w49+3; ss!=w49+4; ss+=1) *rr++ = *ss;
  /* #142: @5 = (-@5) */
  w5 = (- w5 );
  /* #143: (@48[3] = @5) */
  for (rr=w48+3, ss=(&w5); rr!=w48+4; rr+=1) *rr = *ss++;
  /* #144: @5 = @49[2] */
  for (rr=(&w5), ss=w49+2; ss!=w49+3; ss+=1) *rr++ = *ss;
  /* #145: (@48[6] = @5) */
  for (rr=w48+6, ss=(&w5); rr!=w48+7; rr+=1) *rr = *ss++;
  /* #146: @5 = @49[1] */
  for (rr=(&w5), ss=w49+1; ss!=w49+2; ss+=1) *rr++ = *ss;
  /* #147: @5 = (-@5) */
  w5 = (- w5 );
  /* #148: (@48[7] = @5) */
  for (rr=w48+7, ss=(&w5); rr!=w48+8; rr+=1) *rr = *ss++;
  /* #149: @5 = @49[3] */
  for (rr=(&w5), ss=w49+3; ss!=w49+4; ss+=1) *rr++ = *ss;
  /* #150: (@48[1] = @5) */
  for (rr=w48+1, ss=(&w5); rr!=w48+2; rr+=1) *rr = *ss++;
  /* #151: @5 = @49[2] */
  for (rr=(&w5), ss=w49+2; ss!=w49+3; ss+=1) *rr++ = *ss;
  /* #152: @5 = (-@5) */
  w5 = (- w5 );
  /* #153: (@48[2] = @5) */
  for (rr=w48+2, ss=(&w5); rr!=w48+3; rr+=1) *rr = *ss++;
  /* #154: @5 = @49[1] */
  for (rr=(&w5), ss=w49+1; ss!=w49+2; ss+=1) *rr++ = *ss;
  /* #155: (@48[5] = @5) */
  for (rr=w48+5, ss=(&w5); rr!=w48+6; rr+=1) *rr = *ss++;
  /* #156: @50 = (2.*@48) */
  for (i=0, rr=w50, cs=w48; i<9; ++i) *rr++ = (2.* *cs++ );
  /* #157: @47 = mac(@50,@48,@47) */
  for (i=0, rr=w47; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w50+j, tt=w48+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #158: @46 = (@46+@47) */
  for (i=0, rr=w46, cs=w47; i<9; ++i) (*rr++) += (*cs++);
  /* #159: @5 = @49[0] */
  for (rr=(&w5), ss=w49+0; ss!=w49+1; ss+=1) *rr++ = *ss;
  /* #160: @5 = (2.*@5) */
  w5 = (2.* w5 );
  /* #161: @47 = (@5*@48) */
  for (i=0, rr=w47, cs=w48; i<9; ++i) (*rr++)  = (w5*(*cs++));
  /* #162: @46 = (@46+@47) */
  for (i=0, rr=w46, cs=w47; i<9; ++i) (*rr++) += (*cs++);
  /* #163: @7 = input[0][7] */
  w7 = arg[0] ? arg[0][7] : 0;
  /* #164: @13 = input[0][8] */
  w13 = arg[0] ? arg[0][8] : 0;
  /* #165: @18 = input[0][9] */
  w18 = arg[0] ? arg[0][9] : 0;
  /* #166: @51 = vertcat(@7, @13, @18) */
  rr=w51;
  *rr++ = w7;
  *rr++ = w13;
  *rr++ = w18;
  /* #167: @42 = mac(@46,@51,@42) */
  for (i=0, rr=w42; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w46+j, tt=w51+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #168: @7 = dot(@31, @42) */
  w7 = casadi_dot(3, w31, w42);
  /* #169: @4 = (@4*@7) */
  w4 *= w7;
  /* #170: @0 = (@0-@4) */
  w0 -= w4;
  /* #171: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #172: @38 = @38' */
  /* #173: @52 = zeros(1x4) */
  casadi_clear(w52, 4);
  /* #174: @37 = @37' */
  /* #175: @53 = @45' */
  for (i=0, rr=w53, cs=w45; i<4; ++i) for (j=0; j<4; ++j) rr[i+j*4] = *cs++;
  /* #176: @52 = mac(@37,@53,@52) */
  for (i=0, rr=w52; i<4; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w37+j, tt=w53+i*4; k<4; ++k) *rr += ss[k*1]**tt++;
  /* #177: @52 = @52' */
  /* #178: @38 = (@38+@52) */
  for (i=0, rr=w38, cs=w52; i<4; ++i) (*rr++) += (*cs++);
  /* #179: {@0, @4, @7, @13} = vertsplit(@38) */
  w0 = w38[0];
  w4 = w38[1];
  w7 = w38[2];
  w13 = w38[3];
  /* #180: @34 = @34' */
  /* #181: @42 = zeros(1x3) */
  casadi_clear(w42, 3);
  /* #182: @32 = @32' */
  /* #183: @47 = @36' */
  for (i=0, rr=w47, cs=w36; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #184: @42 = mac(@32,@47,@42) */
  for (i=0, rr=w42; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w32+j, tt=w47+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #185: @42 = @42' */
  /* #186: @34 = (@34+@42) */
  for (i=0, rr=w34, cs=w42; i<3; ++i) (*rr++) += (*cs++);
  /* #187: @18 = dot(@31, @34) */
  w18 = casadi_dot(3, w31, w34);
  /* #188: @34 = (@18*@31) */
  for (i=0, rr=w34, cs=w31; i<3; ++i) (*rr++)  = (w18*(*cs++));
  /* #189: @42 = zeros(3x1) */
  casadi_clear(w42, 3);
  /* #190: @47 = @3' */
  for (i=0, rr=w47, cs=w3; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #191: @1 = @1' */
  /* #192: @32 = zeros(1x3) */
  casadi_clear(w32, 3);
  /* #193: @2 = @2' */
  /* #194: @3 = @33' */
  for (i=0, rr=w3, cs=w33; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #195: @32 = mac(@2,@3,@32) */
  for (i=0, rr=w32; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w2+j, tt=w3+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #196: @32 = @32' */
  /* #197: @1 = (@1+@32) */
  for (i=0, rr=w1, cs=w32; i<3; ++i) (*rr++) += (*cs++);
  /* #198: @42 = mac(@47,@1,@42) */
  for (i=0, rr=w42; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w47+j, tt=w1+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #199: @34 = (@34+@42) */
  for (i=0, rr=w34, cs=w42; i<3; ++i) (*rr++) += (*cs++);
  /* #200: @34 = (-@34) */
  for (i=0, rr=w34, cs=w34; i<3; ++i) *rr++ = (- *cs++ );
  /* #201: {@18, @19, @20} = vertsplit(@34) */
  w18 = w34[0];
  w19 = w34[1];
  w20 = w34[2];
  /* #202: @38 = zeros(4x1) */
  casadi_clear(w38, 4);
  /* #203: @47 = zeros(3x3) */
  casadi_clear(w47, 9);
  /* #204: @21 = -0.0001 */
  w21 = -1.0000000000000000e-04;
  /* #205: @31 = (@21*@31) */
  for (i=0, rr=w31, cs=w31; i<3; ++i) (*rr++)  = (w21*(*cs++));
  /* #206: @51 = @51' */
  /* #207: @47 = mac(@31,@51,@47) */
  for (i=0, rr=w47; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w31+j, tt=w51+i*1; k<1; ++k) *rr += ss[k*3]**tt++;
  /* #208: @21 = dot(@48, @47) */
  w21 = casadi_dot(9, w48, w47);
  /* #209: @21 = (2.*@21) */
  w21 = (2.* w21 );
  /* #210: (@38[0] += @21) */
  for (rr=w38+0, ss=(&w21); rr!=w38+1; rr+=1) *rr += *ss++;
  /* #211: @3 = (@5*@47) */
  for (i=0, rr=w3, cs=w47; i<9; ++i) (*rr++)  = (w5*(*cs++));
  /* #212: @33 = zeros(3x3) */
  casadi_clear(w33, 9);
  /* #213: @36 = @50' */
  for (i=0, rr=w36, cs=w50; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #214: @33 = mac(@36,@47,@33) */
  for (i=0, rr=w33; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w36+j, tt=w47+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #215: @3 = (@3+@33) */
  for (i=0, rr=w3, cs=w33; i<9; ++i) (*rr++) += (*cs++);
  /* #216: @33 = zeros(3x3) */
  casadi_clear(w33, 9);
  /* #217: @36 = @48' */
  for (i=0, rr=w36, cs=w48; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #218: @33 = mac(@47,@36,@33) */
  for (i=0, rr=w33; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w47+j, tt=w36+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #219: @33 = (2.*@33) */
  for (i=0, rr=w33, cs=w33; i<9; ++i) *rr++ = (2.* *cs++ );
  /* #220: @3 = (@3+@33) */
  for (i=0, rr=w3, cs=w33; i<9; ++i) (*rr++) += (*cs++);
  /* #221: @5 = @3[5] */
  for (rr=(&w5), ss=w3+5; ss!=w3+6; ss+=1) *rr++ = *ss;
  /* #222: (@38[1] += @5) */
  for (rr=w38+1, ss=(&w5); rr!=w38+2; rr+=1) *rr += *ss++;
  /* #223: @5 = 0 */
  w5 = 0.;
  /* #224: (@3[5] = @5) */
  for (rr=w3+5, ss=(&w5); rr!=w3+6; rr+=1) *rr = *ss++;
  /* #225: @5 = @3[2] */
  for (rr=(&w5), ss=w3+2; ss!=w3+3; ss+=1) *rr++ = *ss;
  /* #226: @5 = (-@5) */
  w5 = (- w5 );
  /* #227: (@38[2] += @5) */
  for (rr=w38+2, ss=(&w5); rr!=w38+3; rr+=1) *rr += *ss++;
  /* #228: @5 = 0 */
  w5 = 0.;
  /* #229: (@3[2] = @5) */
  for (rr=w3+2, ss=(&w5); rr!=w3+3; rr+=1) *rr = *ss++;
  /* #230: @5 = @3[1] */
  for (rr=(&w5), ss=w3+1; ss!=w3+2; ss+=1) *rr++ = *ss;
  /* #231: (@38[3] += @5) */
  for (rr=w38+3, ss=(&w5); rr!=w38+4; rr+=1) *rr += *ss++;
  /* #232: @5 = 0 */
  w5 = 0.;
  /* #233: (@3[1] = @5) */
  for (rr=w3+1, ss=(&w5); rr!=w3+2; rr+=1) *rr = *ss++;
  /* #234: @5 = @3[7] */
  for (rr=(&w5), ss=w3+7; ss!=w3+8; ss+=1) *rr++ = *ss;
  /* #235: @5 = (-@5) */
  w5 = (- w5 );
  /* #236: (@38[1] += @5) */
  for (rr=w38+1, ss=(&w5); rr!=w38+2; rr+=1) *rr += *ss++;
  /* #237: @5 = 0 */
  w5 = 0.;
  /* #238: (@3[7] = @5) */
  for (rr=w3+7, ss=(&w5); rr!=w3+8; rr+=1) *rr = *ss++;
  /* #239: @5 = @3[6] */
  for (rr=(&w5), ss=w3+6; ss!=w3+7; ss+=1) *rr++ = *ss;
  /* #240: (@38[2] += @5) */
  for (rr=w38+2, ss=(&w5); rr!=w38+3; rr+=1) *rr += *ss++;
  /* #241: @5 = 0 */
  w5 = 0.;
  /* #242: (@3[6] = @5) */
  for (rr=w3+6, ss=(&w5); rr!=w3+7; rr+=1) *rr = *ss++;
  /* #243: @5 = @3[3] */
  for (rr=(&w5), ss=w3+3; ss!=w3+4; ss+=1) *rr++ = *ss;
  /* #244: @5 = (-@5) */
  w5 = (- w5 );
  /* #245: (@38[3] += @5) */
  for (rr=w38+3, ss=(&w5); rr!=w38+4; rr+=1) *rr += *ss++;
  /* #246: @52 = (@38/@9) */
  for (i=0, rr=w52, cr=w38; i<4; ++i) (*rr++)  = ((*cr++)/w9);
  /* #247: @49 = (@49/@9) */
  for (i=0, rr=w49; i<4; ++i) (*rr++) /= w9;
  /* #248: @49 = (-@49) */
  for (i=0, rr=w49, cs=w49; i<4; ++i) *rr++ = (- *cs++ );
  /* #249: @5 = dot(@49, @38) */
  w5 = casadi_dot(4, w49, w38);
  /* #250: @5 = (@5/@9) */
  w5 /= w9;
  /* #251: @44 = (@5*@44) */
  for (i=0, rr=w44, cs=w44; i<4; ++i) (*rr++)  = (w5*(*cs++));
  /* #252: @52 = (@52+@44) */
  for (i=0, rr=w52, cs=w44; i<4; ++i) (*rr++) += (*cs++);
  /* #253: {@5, @9, @21, @22} = vertsplit(@52) */
  w5 = w52[0];
  w9 = w52[1];
  w21 = w52[2];
  w22 = w52[3];
  /* #254: @23 = 1 */
  w23 = 1.;
  /* #255: @10 = (@10?@23:0) */
  w10  = (w10?w23:0);
  /* #256: @52 = zeros(4x1) */
  casadi_clear(w52, 4);
  /* #257: @23 = sq(@15) */
  w23 = casadi_sq( w15 );
  /* #258: @24 = sq(@16) */
  w24 = casadi_sq( w16 );
  /* #259: @23 = (@23+@24) */
  w23 += w24;
  /* #260: @24 = (@15/@23) */
  w24  = (w15/w23);
  /* #261: @35 = @35' */
  /* #262: @51 = zeros(1x3) */
  casadi_clear(w51, 3);
  /* #263: @34 = @41' */
  casadi_copy(w41, 3, w34);
  /* #264: @3 = @43' */
  for (i=0, rr=w3, cs=w43; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #265: @51 = mac(@34,@3,@51) */
  for (i=0, rr=w51; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w34+j, tt=w3+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #266: @51 = @51' */
  /* #267: @35 = (@35+@51) */
  for (i=0, rr=w35, cs=w51; i<3; ++i) (*rr++) += (*cs++);
  /* #268: @51 = (@35/@15) */
  for (i=0, rr=w51, cr=w35; i<3; ++i) (*rr++)  = ((*cr++)/w15);
  /* #269: @25 = dot(@40, @51) */
  w25 = casadi_dot(3, w40, w51);
  /* #270: @24 = (@24*@25) */
  w24 *= w25;
  /* #271: @24 = (-@24) */
  w24 = (- w24 );
  /* #272: (@52[0] += @24) */
  for (rr=w52+0, ss=(&w24); rr!=w52+1; rr+=1) *rr += *ss++;
  /* #273: @41 = (@41/@15) */
  for (i=0, rr=w41; i<3; ++i) (*rr++) /= w15;
  /* #274: @41 = (-@41) */
  for (i=0, rr=w41, cs=w41; i<3; ++i) *rr++ = (- *cs++ );
  /* #275: @24 = dot(@41, @35) */
  w24 = casadi_dot(3, w41, w35);
  /* #276: @16 = (@16/@23) */
  w16 /= w23;
  /* #277: @16 = (@16*@25) */
  w16 *= w25;
  /* #278: @24 = (@24+@16) */
  w24 += w16;
  /* #279: @24 = (@24/@15) */
  w24 /= w15;
  /* #280: @39 = (@24*@39) */
  for (i=0, rr=w39, cs=w39; i<3; ++i) (*rr++)  = (w24*(*cs++));
  /* #281: @51 = (@17*@51) */
  for (i=0, rr=w51, cs=w51; i<3; ++i) (*rr++)  = (w17*(*cs++));
  /* #282: @51 = (2.*@51) */
  for (i=0, rr=w51, cs=w51; i<3; ++i) *rr++ = (2.* *cs++ );
  /* #283: @39 = (@39+@51) */
  for (i=0, rr=w39, cs=w51; i<3; ++i) (*rr++) += (*cs++);
  /* #284: (@52[1:4] += @39) */
  for (rr=w52+1, ss=w39; rr!=w52+4; rr+=1) *rr += *ss++;
  /* #285: @44 = (@10*@52) */
  for (i=0, rr=w44, cs=w52; i<4; ++i) (*rr++)  = (w10*(*cs++));
  /* #286: @10 = 1 */
  w10 = 1.;
  /* #287: @14 = (@14?@10:0) */
  w14  = (w14?w10:0);
  /* #288: @52 = (@14*@52) */
  for (i=0, rr=w52, cs=w52; i<4; ++i) (*rr++)  = (w14*(*cs++));
  /* #289: @44 = (@44-@52) */
  for (i=0, rr=w44, cs=w52; i<4; ++i) (*rr++) -= (*cs++);
  /* #290: {@14, @10, @17, @24} = vertsplit(@44) */
  w14 = w44[0];
  w10 = w44[1];
  w17 = w44[2];
  w24 = w44[3];
  /* #291: @15 = (@12*@24) */
  w15  = (w12*w24);
  /* #292: @5 = (@5+@15) */
  w5 += w15;
  /* #293: @15 = (@6*@17) */
  w15  = (w6*w17);
  /* #294: @5 = (@5+@15) */
  w5 += w15;
  /* #295: @15 = (@11*@10) */
  w15  = (w11*w10);
  /* #296: @5 = (@5+@15) */
  w5 += w15;
  /* #297: @15 = (@8*@14) */
  w15  = (w8*w14);
  /* #298: @5 = (@5+@15) */
  w5 += w15;
  /* #299: @15 = (@6*@24) */
  w15  = (w6*w24);
  /* #300: @9 = (@9-@15) */
  w9 -= w15;
  /* #301: @15 = (@12*@17) */
  w15  = (w12*w17);
  /* #302: @9 = (@9+@15) */
  w9 += w15;
  /* #303: @15 = (@8*@10) */
  w15  = (w8*w10);
  /* #304: @9 = (@9+@15) */
  w9 += w15;
  /* #305: @15 = (@11*@14) */
  w15  = (w11*w14);
  /* #306: @9 = (@9-@15) */
  w9 -= w15;
  /* #307: @15 = (@11*@24) */
  w15  = (w11*w24);
  /* #308: @21 = (@21+@15) */
  w21 += w15;
  /* #309: @15 = (@8*@17) */
  w15  = (w8*w17);
  /* #310: @21 = (@21+@15) */
  w21 += w15;
  /* #311: @15 = (@12*@10) */
  w15  = (w12*w10);
  /* #312: @21 = (@21-@15) */
  w21 -= w15;
  /* #313: @15 = (@6*@14) */
  w15  = (w6*w14);
  /* #314: @21 = (@21-@15) */
  w21 -= w15;
  /* #315: @8 = (@8*@24) */
  w8 *= w24;
  /* #316: @22 = (@22+@8) */
  w22 += w8;
  /* #317: @11 = (@11*@17) */
  w11 *= w17;
  /* #318: @22 = (@22-@11) */
  w22 -= w11;
  /* #319: @6 = (@6*@10) */
  w6 *= w10;
  /* #320: @22 = (@22+@6) */
  w22 += w6;
  /* #321: @12 = (@12*@14) */
  w12 *= w14;
  /* #322: @22 = (@22-@12) */
  w22 -= w12;
  /* #323: @39 = zeros(3x1) */
  casadi_clear(w39, 3);
  /* #324: @3 = @46' */
  for (i=0, rr=w3, cs=w46; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #325: @39 = mac(@3,@31,@39) */
  for (i=0, rr=w39; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w3+j, tt=w31+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #326: {@12, @14, @6} = vertsplit(@39) */
  w12 = w39[0];
  w14 = w39[1];
  w6 = w39[2];
  /* #327: @54 = 00 */
  /* #328: @55 = vertcat(@0, @4, @7, @13, @18, @19, @20, @5, @9, @21, @22, @12, @14, @6, @54) */
  rr=w55;
  *rr++ = w0;
  *rr++ = w4;
  *rr++ = w7;
  *rr++ = w13;
  *rr++ = w18;
  *rr++ = w19;
  *rr++ = w20;
  *rr++ = w5;
  *rr++ = w9;
  *rr++ = w21;
  *rr++ = w22;
  *rr++ = w12;
  *rr++ = w14;
  *rr++ = w6;
  /* #329: @56 = dense(@55) */
  casadi_densify(w55, casadi_s1, w56, 0);
  /* #330: output[1][0] = @56 */
  casadi_copy(w56, 15, res[1]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_0_fun_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_0_fun_jac_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_0_fun_jac_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_cost_ext_cost_0_fun_jac_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_0_fun_jac_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_0_fun_jac_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_0_fun_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_0_fun_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s6;
    case 1: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_0_fun_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 29;
  if (sz_res) *sz_res = 6;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 241;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
