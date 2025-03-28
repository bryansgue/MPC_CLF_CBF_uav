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
  #define CASADI_PREFIX(ID) Drone_ode_cost_ext_cost_e_fun_jac_ ## ID
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
static const casadi_int casadi_s1[14] = {11, 1, 0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
static const casadi_int casadi_s2[15] = {11, 1, 0, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
static const casadi_int casadi_s3[3] = {0, 0, 0};
static const casadi_int casadi_s4[29] = {25, 1, 0, 25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
static const casadi_int casadi_s5[5] = {1, 1, 0, 1, 0};

static const casadi_real casadi_c0[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
static const casadi_real casadi_c1[9] = {2., 0., 0., 0., 2., 0., 0., 0., 2.};

/* Drone_ode_cost_ext_cost_e_fun_jac:(i0[11],i1[],i2[],i3[25])->(o0,o1[11]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_int *cii;
  const casadi_real *cr, *cs;
  casadi_real w0, *w1=w+4, *w2=w+7, *w3=w+10, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, w27, w28, w29, *w30=w+45, *w31=w+70, *w32=w+73, *w33=w+76, *w34=w+85, *w35=w+88, *w36=w+91, *w37=w+100, *w38=w+104, *w39=w+108, *w40=w+111, *w41=w+114, *w42=w+117, *w43=w+120, *w44=w+129, *w45=w+138, *w46=w+147, *w47=w+156, *w48=w+165, *w49=w+168, *w50=w+172, *w52=w+176, *w53=w+186;
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
  /* #121: @4 = 0.0001 */
  w4 = 1.0000000000000000e-04;
  /* #122: @42 = zeros(3x1) */
  casadi_clear(w42, 3);
  /* #123: @44 = 
  [[1, 0, 0], 
   [0, 1, 0], 
   [0, 0, 1]] */
  casadi_copy(casadi_c0, 9, w44);
  /* #124: @45 = zeros(3x3) */
  casadi_clear(w45, 9);
  /* #125: @46 = zeros(3x3) */
  casadi_clear(w46, 9);
  /* #126: @38 = vertcat(@9, @5, @7, @13) */
  rr=w38;
  *rr++ = w9;
  *rr++ = w5;
  *rr++ = w7;
  *rr++ = w13;
  /* #127: @9 = ||@38||_F */
  w9 = sqrt(casadi_dot(4, w38, w38));
  /* #128: @37 = (@38/@9) */
  for (i=0, rr=w37, cr=w38; i<4; ++i) (*rr++)  = ((*cr++)/w9);
  /* #129: @5 = @37[3] */
  for (rr=(&w5), ss=w37+3; ss!=w37+4; ss+=1) *rr++ = *ss;
  /* #130: @5 = (-@5) */
  w5 = (- w5 );
  /* #131: (@46[3] = @5) */
  for (rr=w46+3, ss=(&w5); rr!=w46+4; rr+=1) *rr = *ss++;
  /* #132: @5 = @37[2] */
  for (rr=(&w5), ss=w37+2; ss!=w37+3; ss+=1) *rr++ = *ss;
  /* #133: (@46[6] = @5) */
  for (rr=w46+6, ss=(&w5); rr!=w46+7; rr+=1) *rr = *ss++;
  /* #134: @5 = @37[1] */
  for (rr=(&w5), ss=w37+1; ss!=w37+2; ss+=1) *rr++ = *ss;
  /* #135: @5 = (-@5) */
  w5 = (- w5 );
  /* #136: (@46[7] = @5) */
  for (rr=w46+7, ss=(&w5); rr!=w46+8; rr+=1) *rr = *ss++;
  /* #137: @5 = @37[3] */
  for (rr=(&w5), ss=w37+3; ss!=w37+4; ss+=1) *rr++ = *ss;
  /* #138: (@46[1] = @5) */
  for (rr=w46+1, ss=(&w5); rr!=w46+2; rr+=1) *rr = *ss++;
  /* #139: @5 = @37[2] */
  for (rr=(&w5), ss=w37+2; ss!=w37+3; ss+=1) *rr++ = *ss;
  /* #140: @5 = (-@5) */
  w5 = (- w5 );
  /* #141: (@46[2] = @5) */
  for (rr=w46+2, ss=(&w5); rr!=w46+3; rr+=1) *rr = *ss++;
  /* #142: @5 = @37[1] */
  for (rr=(&w5), ss=w37+1; ss!=w37+2; ss+=1) *rr++ = *ss;
  /* #143: (@46[5] = @5) */
  for (rr=w46+5, ss=(&w5); rr!=w46+6; rr+=1) *rr = *ss++;
  /* #144: @47 = (2.*@46) */
  for (i=0, rr=w47, cs=w46; i<9; ++i) *rr++ = (2.* *cs++ );
  /* #145: @45 = mac(@47,@46,@45) */
  for (i=0, rr=w45; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w47+j, tt=w46+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #146: @44 = (@44+@45) */
  for (i=0, rr=w44, cs=w45; i<9; ++i) (*rr++) += (*cs++);
  /* #147: @5 = @37[0] */
  for (rr=(&w5), ss=w37+0; ss!=w37+1; ss+=1) *rr++ = *ss;
  /* #148: @5 = (2.*@5) */
  w5 = (2.* w5 );
  /* #149: @45 = (@5*@46) */
  for (i=0, rr=w45, cs=w46; i<9; ++i) (*rr++)  = (w5*(*cs++));
  /* #150: @44 = (@44+@45) */
  for (i=0, rr=w44, cs=w45; i<9; ++i) (*rr++) += (*cs++);
  /* #151: @7 = input[0][7] */
  w7 = arg[0] ? arg[0][7] : 0;
  /* #152: @13 = input[0][8] */
  w13 = arg[0] ? arg[0][8] : 0;
  /* #153: @18 = input[0][9] */
  w18 = arg[0] ? arg[0][9] : 0;
  /* #154: @48 = vertcat(@7, @13, @18) */
  rr=w48;
  *rr++ = w7;
  *rr++ = w13;
  *rr++ = w18;
  /* #155: @42 = mac(@44,@48,@42) */
  for (i=0, rr=w42; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w44+j, tt=w48+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #156: @7 = dot(@31, @42) */
  w7 = casadi_dot(3, w31, w42);
  /* #157: @4 = (@4*@7) */
  w4 *= w7;
  /* #158: @0 = (@0-@4) */
  w0 -= w4;
  /* #159: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #160: @34 = @34' */
  /* #161: @42 = zeros(1x3) */
  casadi_clear(w42, 3);
  /* #162: @32 = @32' */
  /* #163: @45 = @36' */
  for (i=0, rr=w45, cs=w36; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #164: @42 = mac(@32,@45,@42) */
  for (i=0, rr=w42; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w32+j, tt=w45+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #165: @42 = @42' */
  /* #166: @34 = (@34+@42) */
  for (i=0, rr=w34, cs=w42; i<3; ++i) (*rr++) += (*cs++);
  /* #167: @0 = dot(@31, @34) */
  w0 = casadi_dot(3, w31, w34);
  /* #168: @34 = (@0*@31) */
  for (i=0, rr=w34, cs=w31; i<3; ++i) (*rr++)  = (w0*(*cs++));
  /* #169: @42 = zeros(3x1) */
  casadi_clear(w42, 3);
  /* #170: @45 = @3' */
  for (i=0, rr=w45, cs=w3; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #171: @1 = @1' */
  /* #172: @32 = zeros(1x3) */
  casadi_clear(w32, 3);
  /* #173: @2 = @2' */
  /* #174: @3 = @33' */
  for (i=0, rr=w3, cs=w33; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #175: @32 = mac(@2,@3,@32) */
  for (i=0, rr=w32; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w2+j, tt=w3+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #176: @32 = @32' */
  /* #177: @1 = (@1+@32) */
  for (i=0, rr=w1, cs=w32; i<3; ++i) (*rr++) += (*cs++);
  /* #178: @42 = mac(@45,@1,@42) */
  for (i=0, rr=w42; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w45+j, tt=w1+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #179: @34 = (@34+@42) */
  for (i=0, rr=w34, cs=w42; i<3; ++i) (*rr++) += (*cs++);
  /* #180: @34 = (-@34) */
  for (i=0, rr=w34, cs=w34; i<3; ++i) *rr++ = (- *cs++ );
  /* #181: {@0, @4, @7} = vertsplit(@34) */
  w0 = w34[0];
  w4 = w34[1];
  w7 = w34[2];
  /* #182: @49 = zeros(4x1) */
  casadi_clear(w49, 4);
  /* #183: @45 = zeros(3x3) */
  casadi_clear(w45, 9);
  /* #184: @13 = -0.0001 */
  w13 = -1.0000000000000000e-04;
  /* #185: @31 = (@13*@31) */
  for (i=0, rr=w31, cs=w31; i<3; ++i) (*rr++)  = (w13*(*cs++));
  /* #186: @48 = @48' */
  /* #187: @45 = mac(@31,@48,@45) */
  for (i=0, rr=w45; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w31+j, tt=w48+i*1; k<1; ++k) *rr += ss[k*3]**tt++;
  /* #188: @13 = dot(@46, @45) */
  w13 = casadi_dot(9, w46, w45);
  /* #189: @13 = (2.*@13) */
  w13 = (2.* w13 );
  /* #190: (@49[0] += @13) */
  for (rr=w49+0, ss=(&w13); rr!=w49+1; rr+=1) *rr += *ss++;
  /* #191: @3 = (@5*@45) */
  for (i=0, rr=w3, cs=w45; i<9; ++i) (*rr++)  = (w5*(*cs++));
  /* #192: @33 = zeros(3x3) */
  casadi_clear(w33, 9);
  /* #193: @36 = @47' */
  for (i=0, rr=w36, cs=w47; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #194: @33 = mac(@36,@45,@33) */
  for (i=0, rr=w33; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w36+j, tt=w45+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #195: @3 = (@3+@33) */
  for (i=0, rr=w3, cs=w33; i<9; ++i) (*rr++) += (*cs++);
  /* #196: @33 = zeros(3x3) */
  casadi_clear(w33, 9);
  /* #197: @36 = @46' */
  for (i=0, rr=w36, cs=w46; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #198: @33 = mac(@45,@36,@33) */
  for (i=0, rr=w33; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w45+j, tt=w36+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #199: @33 = (2.*@33) */
  for (i=0, rr=w33, cs=w33; i<9; ++i) *rr++ = (2.* *cs++ );
  /* #200: @3 = (@3+@33) */
  for (i=0, rr=w3, cs=w33; i<9; ++i) (*rr++) += (*cs++);
  /* #201: @5 = @3[5] */
  for (rr=(&w5), ss=w3+5; ss!=w3+6; ss+=1) *rr++ = *ss;
  /* #202: (@49[1] += @5) */
  for (rr=w49+1, ss=(&w5); rr!=w49+2; rr+=1) *rr += *ss++;
  /* #203: @5 = 0 */
  w5 = 0.;
  /* #204: (@3[5] = @5) */
  for (rr=w3+5, ss=(&w5); rr!=w3+6; rr+=1) *rr = *ss++;
  /* #205: @5 = @3[2] */
  for (rr=(&w5), ss=w3+2; ss!=w3+3; ss+=1) *rr++ = *ss;
  /* #206: @5 = (-@5) */
  w5 = (- w5 );
  /* #207: (@49[2] += @5) */
  for (rr=w49+2, ss=(&w5); rr!=w49+3; rr+=1) *rr += *ss++;
  /* #208: @5 = 0 */
  w5 = 0.;
  /* #209: (@3[2] = @5) */
  for (rr=w3+2, ss=(&w5); rr!=w3+3; rr+=1) *rr = *ss++;
  /* #210: @5 = @3[1] */
  for (rr=(&w5), ss=w3+1; ss!=w3+2; ss+=1) *rr++ = *ss;
  /* #211: (@49[3] += @5) */
  for (rr=w49+3, ss=(&w5); rr!=w49+4; rr+=1) *rr += *ss++;
  /* #212: @5 = 0 */
  w5 = 0.;
  /* #213: (@3[1] = @5) */
  for (rr=w3+1, ss=(&w5); rr!=w3+2; rr+=1) *rr = *ss++;
  /* #214: @5 = @3[7] */
  for (rr=(&w5), ss=w3+7; ss!=w3+8; ss+=1) *rr++ = *ss;
  /* #215: @5 = (-@5) */
  w5 = (- w5 );
  /* #216: (@49[1] += @5) */
  for (rr=w49+1, ss=(&w5); rr!=w49+2; rr+=1) *rr += *ss++;
  /* #217: @5 = 0 */
  w5 = 0.;
  /* #218: (@3[7] = @5) */
  for (rr=w3+7, ss=(&w5); rr!=w3+8; rr+=1) *rr = *ss++;
  /* #219: @5 = @3[6] */
  for (rr=(&w5), ss=w3+6; ss!=w3+7; ss+=1) *rr++ = *ss;
  /* #220: (@49[2] += @5) */
  for (rr=w49+2, ss=(&w5); rr!=w49+3; rr+=1) *rr += *ss++;
  /* #221: @5 = 0 */
  w5 = 0.;
  /* #222: (@3[6] = @5) */
  for (rr=w3+6, ss=(&w5); rr!=w3+7; rr+=1) *rr = *ss++;
  /* #223: @5 = @3[3] */
  for (rr=(&w5), ss=w3+3; ss!=w3+4; ss+=1) *rr++ = *ss;
  /* #224: @5 = (-@5) */
  w5 = (- w5 );
  /* #225: (@49[3] += @5) */
  for (rr=w49+3, ss=(&w5); rr!=w49+4; rr+=1) *rr += *ss++;
  /* #226: @50 = (@49/@9) */
  for (i=0, rr=w50, cr=w49; i<4; ++i) (*rr++)  = ((*cr++)/w9);
  /* #227: @37 = (@37/@9) */
  for (i=0, rr=w37; i<4; ++i) (*rr++) /= w9;
  /* #228: @37 = (-@37) */
  for (i=0, rr=w37, cs=w37; i<4; ++i) *rr++ = (- *cs++ );
  /* #229: @5 = dot(@37, @49) */
  w5 = casadi_dot(4, w37, w49);
  /* #230: @5 = (@5/@9) */
  w5 /= w9;
  /* #231: @38 = (@5*@38) */
  for (i=0, rr=w38, cs=w38; i<4; ++i) (*rr++)  = (w5*(*cs++));
  /* #232: @50 = (@50+@38) */
  for (i=0, rr=w50, cs=w38; i<4; ++i) (*rr++) += (*cs++);
  /* #233: {@5, @9, @13, @18} = vertsplit(@50) */
  w5 = w50[0];
  w9 = w50[1];
  w13 = w50[2];
  w18 = w50[3];
  /* #234: @19 = 1 */
  w19 = 1.;
  /* #235: @10 = (@10?@19:0) */
  w10  = (w10?w19:0);
  /* #236: @50 = zeros(4x1) */
  casadi_clear(w50, 4);
  /* #237: @19 = sq(@15) */
  w19 = casadi_sq( w15 );
  /* #238: @20 = sq(@16) */
  w20 = casadi_sq( w16 );
  /* #239: @19 = (@19+@20) */
  w19 += w20;
  /* #240: @20 = (@15/@19) */
  w20  = (w15/w19);
  /* #241: @35 = @35' */
  /* #242: @48 = zeros(1x3) */
  casadi_clear(w48, 3);
  /* #243: @34 = @41' */
  casadi_copy(w41, 3, w34);
  /* #244: @3 = @43' */
  for (i=0, rr=w3, cs=w43; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #245: @48 = mac(@34,@3,@48) */
  for (i=0, rr=w48; i<3; ++i) for (j=0; j<1; ++j, ++rr) for (k=0, ss=w34+j, tt=w3+i*3; k<3; ++k) *rr += ss[k*1]**tt++;
  /* #246: @48 = @48' */
  /* #247: @35 = (@35+@48) */
  for (i=0, rr=w35, cs=w48; i<3; ++i) (*rr++) += (*cs++);
  /* #248: @48 = (@35/@15) */
  for (i=0, rr=w48, cr=w35; i<3; ++i) (*rr++)  = ((*cr++)/w15);
  /* #249: @21 = dot(@40, @48) */
  w21 = casadi_dot(3, w40, w48);
  /* #250: @20 = (@20*@21) */
  w20 *= w21;
  /* #251: @20 = (-@20) */
  w20 = (- w20 );
  /* #252: (@50[0] += @20) */
  for (rr=w50+0, ss=(&w20); rr!=w50+1; rr+=1) *rr += *ss++;
  /* #253: @41 = (@41/@15) */
  for (i=0, rr=w41; i<3; ++i) (*rr++) /= w15;
  /* #254: @41 = (-@41) */
  for (i=0, rr=w41, cs=w41; i<3; ++i) *rr++ = (- *cs++ );
  /* #255: @20 = dot(@41, @35) */
  w20 = casadi_dot(3, w41, w35);
  /* #256: @16 = (@16/@19) */
  w16 /= w19;
  /* #257: @16 = (@16*@21) */
  w16 *= w21;
  /* #258: @20 = (@20+@16) */
  w20 += w16;
  /* #259: @20 = (@20/@15) */
  w20 /= w15;
  /* #260: @39 = (@20*@39) */
  for (i=0, rr=w39, cs=w39; i<3; ++i) (*rr++)  = (w20*(*cs++));
  /* #261: @48 = (@17*@48) */
  for (i=0, rr=w48, cs=w48; i<3; ++i) (*rr++)  = (w17*(*cs++));
  /* #262: @48 = (2.*@48) */
  for (i=0, rr=w48, cs=w48; i<3; ++i) *rr++ = (2.* *cs++ );
  /* #263: @39 = (@39+@48) */
  for (i=0, rr=w39, cs=w48; i<3; ++i) (*rr++) += (*cs++);
  /* #264: (@50[1:4] += @39) */
  for (rr=w50+1, ss=w39; rr!=w50+4; rr+=1) *rr += *ss++;
  /* #265: @38 = (@10*@50) */
  for (i=0, rr=w38, cs=w50; i<4; ++i) (*rr++)  = (w10*(*cs++));
  /* #266: @10 = 1 */
  w10 = 1.;
  /* #267: @14 = (@14?@10:0) */
  w14  = (w14?w10:0);
  /* #268: @50 = (@14*@50) */
  for (i=0, rr=w50, cs=w50; i<4; ++i) (*rr++)  = (w14*(*cs++));
  /* #269: @38 = (@38-@50) */
  for (i=0, rr=w38, cs=w50; i<4; ++i) (*rr++) -= (*cs++);
  /* #270: {@14, @10, @17, @20} = vertsplit(@38) */
  w14 = w38[0];
  w10 = w38[1];
  w17 = w38[2];
  w20 = w38[3];
  /* #271: @15 = (@12*@20) */
  w15  = (w12*w20);
  /* #272: @5 = (@5+@15) */
  w5 += w15;
  /* #273: @15 = (@6*@17) */
  w15  = (w6*w17);
  /* #274: @5 = (@5+@15) */
  w5 += w15;
  /* #275: @15 = (@11*@10) */
  w15  = (w11*w10);
  /* #276: @5 = (@5+@15) */
  w5 += w15;
  /* #277: @15 = (@8*@14) */
  w15  = (w8*w14);
  /* #278: @5 = (@5+@15) */
  w5 += w15;
  /* #279: @15 = (@6*@20) */
  w15  = (w6*w20);
  /* #280: @9 = (@9-@15) */
  w9 -= w15;
  /* #281: @15 = (@12*@17) */
  w15  = (w12*w17);
  /* #282: @9 = (@9+@15) */
  w9 += w15;
  /* #283: @15 = (@8*@10) */
  w15  = (w8*w10);
  /* #284: @9 = (@9+@15) */
  w9 += w15;
  /* #285: @15 = (@11*@14) */
  w15  = (w11*w14);
  /* #286: @9 = (@9-@15) */
  w9 -= w15;
  /* #287: @15 = (@11*@20) */
  w15  = (w11*w20);
  /* #288: @13 = (@13+@15) */
  w13 += w15;
  /* #289: @15 = (@8*@17) */
  w15  = (w8*w17);
  /* #290: @13 = (@13+@15) */
  w13 += w15;
  /* #291: @15 = (@12*@10) */
  w15  = (w12*w10);
  /* #292: @13 = (@13-@15) */
  w13 -= w15;
  /* #293: @15 = (@6*@14) */
  w15  = (w6*w14);
  /* #294: @13 = (@13-@15) */
  w13 -= w15;
  /* #295: @8 = (@8*@20) */
  w8 *= w20;
  /* #296: @18 = (@18+@8) */
  w18 += w8;
  /* #297: @11 = (@11*@17) */
  w11 *= w17;
  /* #298: @18 = (@18-@11) */
  w18 -= w11;
  /* #299: @6 = (@6*@10) */
  w6 *= w10;
  /* #300: @18 = (@18+@6) */
  w18 += w6;
  /* #301: @12 = (@12*@14) */
  w12 *= w14;
  /* #302: @18 = (@18-@12) */
  w18 -= w12;
  /* #303: @39 = zeros(3x1) */
  casadi_clear(w39, 3);
  /* #304: @3 = @44' */
  for (i=0, rr=w3, cs=w44; i<3; ++i) for (j=0; j<3; ++j) rr[i+j*3] = *cs++;
  /* #305: @39 = mac(@3,@31,@39) */
  for (i=0, rr=w39; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w3+j, tt=w31+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #306: {@12, @14, @6} = vertsplit(@39) */
  w12 = w39[0];
  w14 = w39[1];
  w6 = w39[2];
  /* #307: @51 = 00 */
  /* #308: @52 = vertcat(@0, @4, @7, @5, @9, @13, @18, @12, @14, @6, @51) */
  rr=w52;
  *rr++ = w0;
  *rr++ = w4;
  *rr++ = w7;
  *rr++ = w5;
  *rr++ = w9;
  *rr++ = w13;
  *rr++ = w18;
  *rr++ = w12;
  *rr++ = w14;
  *rr++ = w6;
  /* #309: @53 = dense(@52) */
  casadi_densify(w52, casadi_s1, w53, 0);
  /* #310: output[1][0] = @53 */
  casadi_copy(w53, 11, res[1]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_cost_ext_cost_e_fun_jac_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_e_fun_jac_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_cost_ext_cost_e_fun_jac_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_cost_ext_cost_e_fun_jac_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_e_fun_jac_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_cost_ext_cost_e_fun_jac_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_e_fun_jac_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s3;
    case 2: return casadi_s3;
    case 3: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_cost_ext_cost_e_fun_jac_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    case 1: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_cost_ext_cost_e_fun_jac_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 29;
  if (sz_res) *sz_res = 6;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 197;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
