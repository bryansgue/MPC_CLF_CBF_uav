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
  #define CASADI_PREFIX(ID) Drone_ode_complete_2_expl_ode_fun_ ## ID
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
#define casadi_c3 CASADI_PREFIX(c3)
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_mtimes CASADI_PREFIX(mtimes)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)

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

void casadi_mtimes(const casadi_real* x, const casadi_int* sp_x, const casadi_real* y, const casadi_int* sp_y, casadi_real* z, const casadi_int* sp_z, casadi_real* w, casadi_int tr) {
  casadi_int ncol_x, ncol_y, ncol_z, cc;
  const casadi_int *colind_x, *row_x, *colind_y, *row_y, *colind_z, *row_z;
  ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y + 2 + ncol_y+1;
  ncol_z = sp_z[1];
  colind_z = sp_z+2; row_z = sp_z + 2 + ncol_z+1;
  if (tr) {
    for (cc=0; cc<ncol_z; ++cc) {
      casadi_int kk;
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        w[row_y[kk]] = y[kk];
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_z[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          z[kk] += x[kk1] * w[row_x[kk1]];
        }
      }
    }
  } else {
    for (cc=0; cc<ncol_y; ++cc) {
      casadi_int kk;
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        w[row_z[kk]] = z[kk];
      }
      for (kk=colind_y[cc]; kk<colind_y[cc+1]; ++kk) {
        casadi_int kk1;
        casadi_int rr = row_y[kk];
        for (kk1=colind_x[rr]; kk1<colind_x[rr+1]; ++kk1) {
          w[row_x[kk1]] += x[kk1]*y[kk];
        }
      }
      for (kk=colind_z[cc]; kk<colind_z[cc+1]; ++kk) {
        z[kk] = w[row_z[kk]];
      }
    }
  }
}

static const casadi_int casadi_s0[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s1[9] = {3, 3, 0, 1, 2, 3, 0, 1, 2};
static const casadi_int casadi_s2[17] = {13, 1, 0, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
static const casadi_int casadi_s3[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s4[24] = {20, 1, 0, 20, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

static const casadi_real casadi_c0[9] = {1., 0., 0., 0., 1., 0., 0., 0., 1.};
static const casadi_real casadi_c1[3] = {0., 0., 1.};
static const casadi_real casadi_c2[3] = {3.2723905139943781e+02, 6.2619368170575160e+02, 6.2622505275946071e+02};
static const casadi_real casadi_c3[9] = {3.0558700000000000e-03, 0., 0., 0., 1.5969500000000000e-03, 0., 0., 0., 1.5968700000000000e-03};

/* Drone_ode_complete_2_expl_ode_fun:(i0[13],i1[4],i2[20])->(o0[13]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i, j, k;
  casadi_real *rr, *ss, *tt;
  const casadi_real *cs;
  casadi_real w0, *w1=w+4, *w2=w+7, *w3=w+16, *w4=w+25, w5, w6, w7, *w8=w+37, w9, *w10=w+42, w11, w12, *w13=w+53, w14, w15, w16, w17, w18, *w19=w+61, *w20=w+64, *w21=w+67;
  /* #0: @0 = input[0][3] */
  w0 = arg[0] ? arg[0][3] : 0;
  /* #1: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #2: @0 = input[0][4] */
  w0 = arg[0] ? arg[0][4] : 0;
  /* #3: output[0][1] = @0 */
  if (res[0]) res[0][1] = w0;
  /* #4: @0 = input[0][5] */
  w0 = arg[0] ? arg[0][5] : 0;
  /* #5: output[0][2] = @0 */
  if (res[0]) res[0][2] = w0;
  /* #6: @1 = zeros(3x1) */
  casadi_clear(w1, 3);
  /* #7: @2 = 
  [[1, 0, 0], 
   [0, 1, 0], 
   [0, 0, 1]] */
  casadi_copy(casadi_c0, 9, w2);
  /* #8: @3 = zeros(3x3) */
  casadi_clear(w3, 9);
  /* #9: @4 = zeros(3x3) */
  casadi_clear(w4, 9);
  /* #10: @0 = input[0][6] */
  w0 = arg[0] ? arg[0][6] : 0;
  /* #11: @5 = input[0][7] */
  w5 = arg[0] ? arg[0][7] : 0;
  /* #12: @6 = input[0][8] */
  w6 = arg[0] ? arg[0][8] : 0;
  /* #13: @7 = input[0][9] */
  w7 = arg[0] ? arg[0][9] : 0;
  /* #14: @8 = vertcat(@0, @5, @6, @7) */
  rr=w8;
  *rr++ = w0;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  /* #15: @9 = ||@8||_F */
  w9 = sqrt(casadi_dot(4, w8, w8));
  /* #16: @8 = (@8/@9) */
  for (i=0, rr=w8; i<4; ++i) (*rr++) /= w9;
  /* #17: @9 = @8[3] */
  for (rr=(&w9), ss=w8+3; ss!=w8+4; ss+=1) *rr++ = *ss;
  /* #18: @9 = (-@9) */
  w9 = (- w9 );
  /* #19: (@4[3] = @9) */
  for (rr=w4+3, ss=(&w9); rr!=w4+4; rr+=1) *rr = *ss++;
  /* #20: @9 = @8[2] */
  for (rr=(&w9), ss=w8+2; ss!=w8+3; ss+=1) *rr++ = *ss;
  /* #21: (@4[6] = @9) */
  for (rr=w4+6, ss=(&w9); rr!=w4+7; rr+=1) *rr = *ss++;
  /* #22: @9 = @8[1] */
  for (rr=(&w9), ss=w8+1; ss!=w8+2; ss+=1) *rr++ = *ss;
  /* #23: @9 = (-@9) */
  w9 = (- w9 );
  /* #24: (@4[7] = @9) */
  for (rr=w4+7, ss=(&w9); rr!=w4+8; rr+=1) *rr = *ss++;
  /* #25: @9 = @8[3] */
  for (rr=(&w9), ss=w8+3; ss!=w8+4; ss+=1) *rr++ = *ss;
  /* #26: (@4[1] = @9) */
  for (rr=w4+1, ss=(&w9); rr!=w4+2; rr+=1) *rr = *ss++;
  /* #27: @9 = @8[2] */
  for (rr=(&w9), ss=w8+2; ss!=w8+3; ss+=1) *rr++ = *ss;
  /* #28: @9 = (-@9) */
  w9 = (- w9 );
  /* #29: (@4[2] = @9) */
  for (rr=w4+2, ss=(&w9); rr!=w4+3; rr+=1) *rr = *ss++;
  /* #30: @9 = @8[1] */
  for (rr=(&w9), ss=w8+1; ss!=w8+2; ss+=1) *rr++ = *ss;
  /* #31: (@4[5] = @9) */
  for (rr=w4+5, ss=(&w9); rr!=w4+6; rr+=1) *rr = *ss++;
  /* #32: @10 = (2.*@4) */
  for (i=0, rr=w10, cs=w4; i<9; ++i) *rr++ = (2.* *cs++ );
  /* #33: @3 = mac(@10,@4,@3) */
  for (i=0, rr=w3; i<3; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w10+j, tt=w4+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #34: @2 = (@2+@3) */
  for (i=0, rr=w2, cs=w3; i<9; ++i) (*rr++) += (*cs++);
  /* #35: @9 = @8[0] */
  for (rr=(&w9), ss=w8+0; ss!=w8+1; ss+=1) *rr++ = *ss;
  /* #36: @9 = (2.*@9) */
  w9 = (2.* w9 );
  /* #37: @4 = (@9*@4) */
  for (i=0, rr=w4, cs=w4; i<9; ++i) (*rr++)  = (w9*(*cs++));
  /* #38: @2 = (@2+@4) */
  for (i=0, rr=w2, cs=w4; i<9; ++i) (*rr++) += (*cs++);
  /* #39: @9 = 0 */
  w9 = 0.;
  /* #40: @11 = 0 */
  w11 = 0.;
  /* #41: @12 = input[1][0] */
  w12 = arg[1] ? arg[1][0] : 0;
  /* #42: @13 = vertcat(@9, @11, @12) */
  rr=w13;
  *rr++ = w9;
  *rr++ = w11;
  *rr++ = w12;
  /* #43: @1 = mac(@2,@13,@1) */
  for (i=0, rr=w1; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w2+j, tt=w13+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #44: @9 = 0.85 */
  w9 = 8.4999999999999998e-01;
  /* #45: @1 = (@1/@9) */
  for (i=0, rr=w1; i<3; ++i) (*rr++) /= w9;
  /* #46: @13 = [0, 0, 1] */
  casadi_copy(casadi_c1, 3, w13);
  /* #47: @9 = 9.81 */
  w9 = 9.8100000000000005e+00;
  /* #48: @13 = (@13*@9) */
  for (i=0, rr=w13; i<3; ++i) (*rr++) *= w9;
  /* #49: @1 = (@1-@13) */
  for (i=0, rr=w1, cs=w13; i<3; ++i) (*rr++) -= (*cs++);
  /* #50: output[0][3] = @1 */
  if (res[0]) casadi_copy(w1, 3, res[0]+3);
  /* #51: @9 = 0.5 */
  w9 = 5.0000000000000000e-01;
  /* #52: @11 = input[0][10] */
  w11 = arg[0] ? arg[0][10] : 0;
  /* #53: @12 = (@5*@11) */
  w12  = (w5*w11);
  /* #54: @12 = (-@12) */
  w12 = (- w12 );
  /* #55: @14 = input[0][11] */
  w14 = arg[0] ? arg[0][11] : 0;
  /* #56: @15 = (@6*@14) */
  w15  = (w6*w14);
  /* #57: @12 = (@12-@15) */
  w12 -= w15;
  /* #58: @15 = input[0][12] */
  w15 = arg[0] ? arg[0][12] : 0;
  /* #59: @16 = (@7*@15) */
  w16  = (w7*w15);
  /* #60: @12 = (@12-@16) */
  w12 -= w16;
  /* #61: @16 = (@0*@11) */
  w16  = (w0*w11);
  /* #62: @17 = (@6*@15) */
  w17  = (w6*w15);
  /* #63: @16 = (@16+@17) */
  w16 += w17;
  /* #64: @17 = (@7*@14) */
  w17  = (w7*w14);
  /* #65: @16 = (@16-@17) */
  w16 -= w17;
  /* #66: @17 = (@0*@14) */
  w17  = (w0*w14);
  /* #67: @18 = (@5*@15) */
  w18  = (w5*w15);
  /* #68: @17 = (@17-@18) */
  w17 -= w18;
  /* #69: @7 = (@7*@11) */
  w7 *= w11;
  /* #70: @17 = (@17+@7) */
  w17 += w7;
  /* #71: @0 = (@0*@15) */
  w0 *= w15;
  /* #72: @5 = (@5*@14) */
  w5 *= w14;
  /* #73: @0 = (@0+@5) */
  w0 += w5;
  /* #74: @6 = (@6*@11) */
  w6 *= w11;
  /* #75: @0 = (@0-@6) */
  w0 -= w6;
  /* #76: @8 = vertcat(@12, @16, @17, @0) */
  rr=w8;
  *rr++ = w12;
  *rr++ = w16;
  *rr++ = w17;
  *rr++ = w0;
  /* #77: @8 = (@9*@8) */
  for (i=0, rr=w8, cs=w8; i<4; ++i) (*rr++)  = (w9*(*cs++));
  /* #78: output[0][4] = @8 */
  if (res[0]) casadi_copy(w8, 4, res[0]+6);
  /* #79: @1 = zeros(3x1) */
  casadi_clear(w1, 3);
  /* #80: @13 = 
  [[327.239, 00, 00], 
   [00, 626.194, 00], 
   [00, 00, 626.225]] */
  casadi_copy(casadi_c2, 3, w13);
  /* #81: @9 = input[1][1] */
  w9 = arg[1] ? arg[1][1] : 0;
  /* #82: @12 = input[1][2] */
  w12 = arg[1] ? arg[1][2] : 0;
  /* #83: @16 = input[1][3] */
  w16 = arg[1] ? arg[1][3] : 0;
  /* #84: @19 = vertcat(@9, @12, @16) */
  rr=w19;
  *rr++ = w9;
  *rr++ = w12;
  *rr++ = w16;
  /* #85: @20 = zeros(3x1) */
  casadi_clear(w20, 3);
  /* #86: @2 = 
  [[0.00305587, 0, 0], 
   [0, 0.00159695, 0], 
   [0, 0, 0.00159687]] */
  casadi_copy(casadi_c3, 9, w2);
  /* #87: @21 = vertcat(@11, @14, @15) */
  rr=w21;
  *rr++ = w11;
  *rr++ = w14;
  *rr++ = w15;
  /* #88: @20 = mac(@2,@21,@20) */
  for (i=0, rr=w20; i<1; ++i) for (j=0; j<3; ++j, ++rr) for (k=0, ss=w2+j, tt=w21+i*3; k<3; ++k) *rr += ss[k*3]**tt++;
  /* #89: @9 = @20[2] */
  for (rr=(&w9), ss=w20+2; ss!=w20+3; ss+=1) *rr++ = *ss;
  /* #90: @12 = (@14*@9) */
  w12  = (w14*w9);
  /* #91: @16 = @20[1] */
  for (rr=(&w16), ss=w20+1; ss!=w20+2; ss+=1) *rr++ = *ss;
  /* #92: @17 = (@15*@16) */
  w17  = (w15*w16);
  /* #93: @12 = (@12-@17) */
  w12 -= w17;
  /* #94: @17 = @20[0] */
  for (rr=(&w17), ss=w20+0; ss!=w20+1; ss+=1) *rr++ = *ss;
  /* #95: @15 = (@15*@17) */
  w15 *= w17;
  /* #96: @9 = (@11*@9) */
  w9  = (w11*w9);
  /* #97: @15 = (@15-@9) */
  w15 -= w9;
  /* #98: @11 = (@11*@16) */
  w11 *= w16;
  /* #99: @14 = (@14*@17) */
  w14 *= w17;
  /* #100: @11 = (@11-@14) */
  w11 -= w14;
  /* #101: @20 = vertcat(@12, @15, @11) */
  rr=w20;
  *rr++ = w12;
  *rr++ = w15;
  *rr++ = w11;
  /* #102: @19 = (@19-@20) */
  for (i=0, rr=w19, cs=w20; i<3; ++i) (*rr++) -= (*cs++);
  /* #103: @1 = mac(@13,@19,@1) */
  casadi_mtimes(w13, casadi_s1, w19, casadi_s0, w1, casadi_s0, w, 0);
  /* #104: output[0][5] = @1 */
  if (res[0]) casadi_copy(w1, 3, res[0]+10);
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_2_expl_ode_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_2_expl_ode_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_2_expl_ode_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_2_expl_ode_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_2_expl_ode_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_2_expl_ode_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_2_expl_ode_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void Drone_ode_complete_2_expl_ode_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_complete_2_expl_ode_fun_n_in(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_int Drone_ode_complete_2_expl_ode_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real Drone_ode_complete_2_expl_ode_fun_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_complete_2_expl_ode_fun_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Drone_ode_complete_2_expl_ode_fun_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_complete_2_expl_ode_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Drone_ode_complete_2_expl_ode_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Drone_ode_complete_2_expl_ode_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 7;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 70;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
