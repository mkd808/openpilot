#include "car.h"

namespace {
#define DIM 9
#define EDIM 9
#define MEDIM 9
typedef void (*Hfun)(double *, double *, double *);

double mass;

void set_mass(double x){ mass = x;}

double rotational_inertia;

void set_rotational_inertia(double x){ rotational_inertia = x;}

double center_to_front;

void set_center_to_front(double x){ center_to_front = x;}

double center_to_rear;

void set_center_to_rear(double x){ center_to_rear = x;}

double stiffness_front;

void set_stiffness_front(double x){ stiffness_front = x;}

double stiffness_rear;

void set_stiffness_rear(double x){ stiffness_rear = x;}
const static double MAHA_THRESH_25 = 3.8414588206941227;
const static double MAHA_THRESH_24 = 5.991464547107981;
const static double MAHA_THRESH_30 = 3.8414588206941227;
const static double MAHA_THRESH_26 = 3.8414588206941227;
const static double MAHA_THRESH_27 = 3.8414588206941227;
const static double MAHA_THRESH_29 = 3.8414588206941227;
const static double MAHA_THRESH_28 = 3.8414588206941227;
const static double MAHA_THRESH_31 = 3.8414588206941227;

/******************************************************************************
 *                       Code generated with SymPy 1.12                       *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                         This file is part of 'ekf'                         *
 ******************************************************************************/
void err_fun(double *nom_x, double *delta_x, double *out_2290816945354955451) {
   out_2290816945354955451[0] = delta_x[0] + nom_x[0];
   out_2290816945354955451[1] = delta_x[1] + nom_x[1];
   out_2290816945354955451[2] = delta_x[2] + nom_x[2];
   out_2290816945354955451[3] = delta_x[3] + nom_x[3];
   out_2290816945354955451[4] = delta_x[4] + nom_x[4];
   out_2290816945354955451[5] = delta_x[5] + nom_x[5];
   out_2290816945354955451[6] = delta_x[6] + nom_x[6];
   out_2290816945354955451[7] = delta_x[7] + nom_x[7];
   out_2290816945354955451[8] = delta_x[8] + nom_x[8];
}
void inv_err_fun(double *nom_x, double *true_x, double *out_8668625092093883095) {
   out_8668625092093883095[0] = -nom_x[0] + true_x[0];
   out_8668625092093883095[1] = -nom_x[1] + true_x[1];
   out_8668625092093883095[2] = -nom_x[2] + true_x[2];
   out_8668625092093883095[3] = -nom_x[3] + true_x[3];
   out_8668625092093883095[4] = -nom_x[4] + true_x[4];
   out_8668625092093883095[5] = -nom_x[5] + true_x[5];
   out_8668625092093883095[6] = -nom_x[6] + true_x[6];
   out_8668625092093883095[7] = -nom_x[7] + true_x[7];
   out_8668625092093883095[8] = -nom_x[8] + true_x[8];
}
void H_mod_fun(double *state, double *out_4743823478057154038) {
   out_4743823478057154038[0] = 1.0;
   out_4743823478057154038[1] = 0;
   out_4743823478057154038[2] = 0;
   out_4743823478057154038[3] = 0;
   out_4743823478057154038[4] = 0;
   out_4743823478057154038[5] = 0;
   out_4743823478057154038[6] = 0;
   out_4743823478057154038[7] = 0;
   out_4743823478057154038[8] = 0;
   out_4743823478057154038[9] = 0;
   out_4743823478057154038[10] = 1.0;
   out_4743823478057154038[11] = 0;
   out_4743823478057154038[12] = 0;
   out_4743823478057154038[13] = 0;
   out_4743823478057154038[14] = 0;
   out_4743823478057154038[15] = 0;
   out_4743823478057154038[16] = 0;
   out_4743823478057154038[17] = 0;
   out_4743823478057154038[18] = 0;
   out_4743823478057154038[19] = 0;
   out_4743823478057154038[20] = 1.0;
   out_4743823478057154038[21] = 0;
   out_4743823478057154038[22] = 0;
   out_4743823478057154038[23] = 0;
   out_4743823478057154038[24] = 0;
   out_4743823478057154038[25] = 0;
   out_4743823478057154038[26] = 0;
   out_4743823478057154038[27] = 0;
   out_4743823478057154038[28] = 0;
   out_4743823478057154038[29] = 0;
   out_4743823478057154038[30] = 1.0;
   out_4743823478057154038[31] = 0;
   out_4743823478057154038[32] = 0;
   out_4743823478057154038[33] = 0;
   out_4743823478057154038[34] = 0;
   out_4743823478057154038[35] = 0;
   out_4743823478057154038[36] = 0;
   out_4743823478057154038[37] = 0;
   out_4743823478057154038[38] = 0;
   out_4743823478057154038[39] = 0;
   out_4743823478057154038[40] = 1.0;
   out_4743823478057154038[41] = 0;
   out_4743823478057154038[42] = 0;
   out_4743823478057154038[43] = 0;
   out_4743823478057154038[44] = 0;
   out_4743823478057154038[45] = 0;
   out_4743823478057154038[46] = 0;
   out_4743823478057154038[47] = 0;
   out_4743823478057154038[48] = 0;
   out_4743823478057154038[49] = 0;
   out_4743823478057154038[50] = 1.0;
   out_4743823478057154038[51] = 0;
   out_4743823478057154038[52] = 0;
   out_4743823478057154038[53] = 0;
   out_4743823478057154038[54] = 0;
   out_4743823478057154038[55] = 0;
   out_4743823478057154038[56] = 0;
   out_4743823478057154038[57] = 0;
   out_4743823478057154038[58] = 0;
   out_4743823478057154038[59] = 0;
   out_4743823478057154038[60] = 1.0;
   out_4743823478057154038[61] = 0;
   out_4743823478057154038[62] = 0;
   out_4743823478057154038[63] = 0;
   out_4743823478057154038[64] = 0;
   out_4743823478057154038[65] = 0;
   out_4743823478057154038[66] = 0;
   out_4743823478057154038[67] = 0;
   out_4743823478057154038[68] = 0;
   out_4743823478057154038[69] = 0;
   out_4743823478057154038[70] = 1.0;
   out_4743823478057154038[71] = 0;
   out_4743823478057154038[72] = 0;
   out_4743823478057154038[73] = 0;
   out_4743823478057154038[74] = 0;
   out_4743823478057154038[75] = 0;
   out_4743823478057154038[76] = 0;
   out_4743823478057154038[77] = 0;
   out_4743823478057154038[78] = 0;
   out_4743823478057154038[79] = 0;
   out_4743823478057154038[80] = 1.0;
}
void f_fun(double *state, double dt, double *out_3260309363242572689) {
   out_3260309363242572689[0] = state[0];
   out_3260309363242572689[1] = state[1];
   out_3260309363242572689[2] = state[2];
   out_3260309363242572689[3] = state[3];
   out_3260309363242572689[4] = state[4];
   out_3260309363242572689[5] = dt*((-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]))*state[6] - 9.8000000000000007*state[8] + stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*state[1]) + (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*state[4])) + state[5];
   out_3260309363242572689[6] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*state[4])) + state[6];
   out_3260309363242572689[7] = state[7];
   out_3260309363242572689[8] = state[8];
}
void F_fun(double *state, double dt, double *out_7845718988328896002) {
   out_7845718988328896002[0] = 1;
   out_7845718988328896002[1] = 0;
   out_7845718988328896002[2] = 0;
   out_7845718988328896002[3] = 0;
   out_7845718988328896002[4] = 0;
   out_7845718988328896002[5] = 0;
   out_7845718988328896002[6] = 0;
   out_7845718988328896002[7] = 0;
   out_7845718988328896002[8] = 0;
   out_7845718988328896002[9] = 0;
   out_7845718988328896002[10] = 1;
   out_7845718988328896002[11] = 0;
   out_7845718988328896002[12] = 0;
   out_7845718988328896002[13] = 0;
   out_7845718988328896002[14] = 0;
   out_7845718988328896002[15] = 0;
   out_7845718988328896002[16] = 0;
   out_7845718988328896002[17] = 0;
   out_7845718988328896002[18] = 0;
   out_7845718988328896002[19] = 0;
   out_7845718988328896002[20] = 1;
   out_7845718988328896002[21] = 0;
   out_7845718988328896002[22] = 0;
   out_7845718988328896002[23] = 0;
   out_7845718988328896002[24] = 0;
   out_7845718988328896002[25] = 0;
   out_7845718988328896002[26] = 0;
   out_7845718988328896002[27] = 0;
   out_7845718988328896002[28] = 0;
   out_7845718988328896002[29] = 0;
   out_7845718988328896002[30] = 1;
   out_7845718988328896002[31] = 0;
   out_7845718988328896002[32] = 0;
   out_7845718988328896002[33] = 0;
   out_7845718988328896002[34] = 0;
   out_7845718988328896002[35] = 0;
   out_7845718988328896002[36] = 0;
   out_7845718988328896002[37] = 0;
   out_7845718988328896002[38] = 0;
   out_7845718988328896002[39] = 0;
   out_7845718988328896002[40] = 1;
   out_7845718988328896002[41] = 0;
   out_7845718988328896002[42] = 0;
   out_7845718988328896002[43] = 0;
   out_7845718988328896002[44] = 0;
   out_7845718988328896002[45] = dt*(stiffness_front*(-state[2] - state[3] + state[7])/(mass*state[1]) + (-stiffness_front - stiffness_rear)*state[5]/(mass*state[4]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[6]/(mass*state[4]));
   out_7845718988328896002[46] = -dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(mass*pow(state[1], 2));
   out_7845718988328896002[47] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_7845718988328896002[48] = -dt*stiffness_front*state[0]/(mass*state[1]);
   out_7845718988328896002[49] = dt*((-1 - (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*pow(state[4], 2)))*state[6] - (-stiffness_front*state[0] - stiffness_rear*state[0])*state[5]/(mass*pow(state[4], 2)));
   out_7845718988328896002[50] = dt*(-stiffness_front*state[0] - stiffness_rear*state[0])/(mass*state[4]) + 1;
   out_7845718988328896002[51] = dt*(-state[4] + (-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(mass*state[4]));
   out_7845718988328896002[52] = dt*stiffness_front*state[0]/(mass*state[1]);
   out_7845718988328896002[53] = -9.8000000000000007*dt;
   out_7845718988328896002[54] = dt*(center_to_front*stiffness_front*(-state[2] - state[3] + state[7])/(rotational_inertia*state[1]) + (-center_to_front*stiffness_front + center_to_rear*stiffness_rear)*state[5]/(rotational_inertia*state[4]) + (-pow(center_to_front, 2)*stiffness_front - pow(center_to_rear, 2)*stiffness_rear)*state[6]/(rotational_inertia*state[4]));
   out_7845718988328896002[55] = -center_to_front*dt*stiffness_front*(-state[2] - state[3] + state[7])*state[0]/(rotational_inertia*pow(state[1], 2));
   out_7845718988328896002[56] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7845718988328896002[57] = -center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7845718988328896002[58] = dt*(-(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])*state[5]/(rotational_inertia*pow(state[4], 2)) - (-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])*state[6]/(rotational_inertia*pow(state[4], 2)));
   out_7845718988328896002[59] = dt*(-center_to_front*stiffness_front*state[0] + center_to_rear*stiffness_rear*state[0])/(rotational_inertia*state[4]);
   out_7845718988328896002[60] = dt*(-pow(center_to_front, 2)*stiffness_front*state[0] - pow(center_to_rear, 2)*stiffness_rear*state[0])/(rotational_inertia*state[4]) + 1;
   out_7845718988328896002[61] = center_to_front*dt*stiffness_front*state[0]/(rotational_inertia*state[1]);
   out_7845718988328896002[62] = 0;
   out_7845718988328896002[63] = 0;
   out_7845718988328896002[64] = 0;
   out_7845718988328896002[65] = 0;
   out_7845718988328896002[66] = 0;
   out_7845718988328896002[67] = 0;
   out_7845718988328896002[68] = 0;
   out_7845718988328896002[69] = 0;
   out_7845718988328896002[70] = 1;
   out_7845718988328896002[71] = 0;
   out_7845718988328896002[72] = 0;
   out_7845718988328896002[73] = 0;
   out_7845718988328896002[74] = 0;
   out_7845718988328896002[75] = 0;
   out_7845718988328896002[76] = 0;
   out_7845718988328896002[77] = 0;
   out_7845718988328896002[78] = 0;
   out_7845718988328896002[79] = 0;
   out_7845718988328896002[80] = 1;
}
void h_25(double *state, double *unused, double *out_1560465764258318145) {
   out_1560465764258318145[0] = state[6];
}
void H_25(double *state, double *unused, double *out_1652686704141933712) {
   out_1652686704141933712[0] = 0;
   out_1652686704141933712[1] = 0;
   out_1652686704141933712[2] = 0;
   out_1652686704141933712[3] = 0;
   out_1652686704141933712[4] = 0;
   out_1652686704141933712[5] = 0;
   out_1652686704141933712[6] = 1;
   out_1652686704141933712[7] = 0;
   out_1652686704141933712[8] = 0;
}
void h_24(double *state, double *unused, double *out_8424072936813750229) {
   out_8424072936813750229[0] = state[4];
   out_8424072936813750229[1] = state[5];
}
void H_24(double *state, double *unused, double *out_519962894863565854) {
   out_519962894863565854[0] = 0;
   out_519962894863565854[1] = 0;
   out_519962894863565854[2] = 0;
   out_519962894863565854[3] = 0;
   out_519962894863565854[4] = 1;
   out_519962894863565854[5] = 0;
   out_519962894863565854[6] = 0;
   out_519962894863565854[7] = 0;
   out_519962894863565854[8] = 0;
   out_519962894863565854[9] = 0;
   out_519962894863565854[10] = 0;
   out_519962894863565854[11] = 0;
   out_519962894863565854[12] = 0;
   out_519962894863565854[13] = 0;
   out_519962894863565854[14] = 1;
   out_519962894863565854[15] = 0;
   out_519962894863565854[16] = 0;
   out_519962894863565854[17] = 0;
}
void h_30(double *state, double *unused, double *out_6202799492562354296) {
   out_6202799492562354296[0] = state[4];
}
void H_30(double *state, double *unused, double *out_4171019662649182339) {
   out_4171019662649182339[0] = 0;
   out_4171019662649182339[1] = 0;
   out_4171019662649182339[2] = 0;
   out_4171019662649182339[3] = 0;
   out_4171019662649182339[4] = 1;
   out_4171019662649182339[5] = 0;
   out_4171019662649182339[6] = 0;
   out_4171019662649182339[7] = 0;
   out_4171019662649182339[8] = 0;
}
void h_26(double *state, double *unused, double *out_196242024851494850) {
   out_196242024851494850[0] = state[7];
}
void H_26(double *state, double *unused, double *out_2088816614732122512) {
   out_2088816614732122512[0] = 0;
   out_2088816614732122512[1] = 0;
   out_2088816614732122512[2] = 0;
   out_2088816614732122512[3] = 0;
   out_2088816614732122512[4] = 0;
   out_2088816614732122512[5] = 0;
   out_2088816614732122512[6] = 0;
   out_2088816614732122512[7] = 1;
   out_2088816614732122512[8] = 0;
}
void h_27(double *state, double *unused, double *out_8666281197547850443) {
   out_8666281197547850443[0] = state[3];
}
void H_27(double *state, double *unused, double *out_6394613733833125556) {
   out_6394613733833125556[0] = 0;
   out_6394613733833125556[1] = 0;
   out_6394613733833125556[2] = 0;
   out_6394613733833125556[3] = 1;
   out_6394613733833125556[4] = 0;
   out_6394613733833125556[5] = 0;
   out_6394613733833125556[6] = 0;
   out_6394613733833125556[7] = 0;
   out_6394613733833125556[8] = 0;
}
void h_29(double *state, double *unused, double *out_1201126224901508651) {
   out_1201126224901508651[0] = state[1];
}
void H_29(double *state, double *unused, double *out_4681251006963574523) {
   out_4681251006963574523[0] = 0;
   out_4681251006963574523[1] = 1;
   out_4681251006963574523[2] = 0;
   out_4681251006963574523[3] = 0;
   out_4681251006963574523[4] = 0;
   out_4681251006963574523[5] = 0;
   out_4681251006963574523[6] = 0;
   out_4681251006963574523[7] = 0;
   out_4681251006963574523[8] = 0;
}
void h_28(double *state, double *unused, double *out_6818187523072045103) {
   out_6818187523072045103[0] = state[0];
}
void H_28(double *state, double *unused, double *out_401148010105956051) {
   out_401148010105956051[0] = 1;
   out_401148010105956051[1] = 0;
   out_401148010105956051[2] = 0;
   out_401148010105956051[3] = 0;
   out_401148010105956051[4] = 0;
   out_401148010105956051[5] = 0;
   out_401148010105956051[6] = 0;
   out_401148010105956051[7] = 0;
   out_401148010105956051[8] = 0;
}
void h_31(double *state, double *unused, double *out_5171293783944956096) {
   out_5171293783944956096[0] = state[8];
}
void H_31(double *state, double *unused, double *out_1683332666018894140) {
   out_1683332666018894140[0] = 0;
   out_1683332666018894140[1] = 0;
   out_1683332666018894140[2] = 0;
   out_1683332666018894140[3] = 0;
   out_1683332666018894140[4] = 0;
   out_1683332666018894140[5] = 0;
   out_1683332666018894140[6] = 0;
   out_1683332666018894140[7] = 0;
   out_1683332666018894140[8] = 1;
}
#include <eigen3/Eigen/Dense>
#include <iostream>

typedef Eigen::Matrix<double, DIM, DIM, Eigen::RowMajor> DDM;
typedef Eigen::Matrix<double, EDIM, EDIM, Eigen::RowMajor> EEM;
typedef Eigen::Matrix<double, DIM, EDIM, Eigen::RowMajor> DEM;

void predict(double *in_x, double *in_P, double *in_Q, double dt) {
  typedef Eigen::Matrix<double, MEDIM, MEDIM, Eigen::RowMajor> RRM;

  double nx[DIM] = {0};
  double in_F[EDIM*EDIM] = {0};

  // functions from sympy
  f_fun(in_x, dt, nx);
  F_fun(in_x, dt, in_F);


  EEM F(in_F);
  EEM P(in_P);
  EEM Q(in_Q);

  RRM F_main = F.topLeftCorner(MEDIM, MEDIM);
  P.topLeftCorner(MEDIM, MEDIM) = (F_main * P.topLeftCorner(MEDIM, MEDIM)) * F_main.transpose();
  P.topRightCorner(MEDIM, EDIM - MEDIM) = F_main * P.topRightCorner(MEDIM, EDIM - MEDIM);
  P.bottomLeftCorner(EDIM - MEDIM, MEDIM) = P.bottomLeftCorner(EDIM - MEDIM, MEDIM) * F_main.transpose();

  P = P + dt*Q;

  // copy out state
  memcpy(in_x, nx, DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
}

// note: extra_args dim only correct when null space projecting
// otherwise 1
template <int ZDIM, int EADIM, bool MAHA_TEST>
void update(double *in_x, double *in_P, Hfun h_fun, Hfun H_fun, Hfun Hea_fun, double *in_z, double *in_R, double *in_ea, double MAHA_THRESHOLD) {
  typedef Eigen::Matrix<double, ZDIM, ZDIM, Eigen::RowMajor> ZZM;
  typedef Eigen::Matrix<double, ZDIM, DIM, Eigen::RowMajor> ZDM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, EDIM, Eigen::RowMajor> XEM;
  //typedef Eigen::Matrix<double, EDIM, ZDIM, Eigen::RowMajor> EZM;
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> X1M;
  typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> XXM;

  double in_hx[ZDIM] = {0};
  double in_H[ZDIM * DIM] = {0};
  double in_H_mod[EDIM * DIM] = {0};
  double delta_x[EDIM] = {0};
  double x_new[DIM] = {0};


  // state x, P
  Eigen::Matrix<double, ZDIM, 1> z(in_z);
  EEM P(in_P);
  ZZM pre_R(in_R);

  // functions from sympy
  h_fun(in_x, in_ea, in_hx);
  H_fun(in_x, in_ea, in_H);
  ZDM pre_H(in_H);

  // get y (y = z - hx)
  Eigen::Matrix<double, ZDIM, 1> pre_y(in_hx); pre_y = z - pre_y;
  X1M y; XXM H; XXM R;
  if (Hea_fun){
    typedef Eigen::Matrix<double, ZDIM, EADIM, Eigen::RowMajor> ZAM;
    double in_Hea[ZDIM * EADIM] = {0};
    Hea_fun(in_x, in_ea, in_Hea);
    ZAM Hea(in_Hea);
    XXM A = Hea.transpose().fullPivLu().kernel();


    y = A.transpose() * pre_y;
    H = A.transpose() * pre_H;
    R = A.transpose() * pre_R * A;
  } else {
    y = pre_y;
    H = pre_H;
    R = pre_R;
  }
  // get modified H
  H_mod_fun(in_x, in_H_mod);
  DEM H_mod(in_H_mod);
  XEM H_err = H * H_mod;

  // Do mahalobis distance test
  if (MAHA_TEST){
    XXM a = (H_err * P * H_err.transpose() + R).inverse();
    double maha_dist = y.transpose() * a * y;
    if (maha_dist > MAHA_THRESHOLD){
      R = 1.0e16 * R;
    }
  }

  // Outlier resilient weighting
  double weight = 1;//(1.5)/(1 + y.squaredNorm()/R.sum());

  // kalman gains and I_KH
  XXM S = ((H_err * P) * H_err.transpose()) + R/weight;
  XEM KT = S.fullPivLu().solve(H_err * P.transpose());
  //EZM K = KT.transpose(); TODO: WHY DOES THIS NOT COMPILE?
  //EZM K = S.fullPivLu().solve(H_err * P.transpose()).transpose();
  //std::cout << "Here is the matrix rot:\n" << K << std::endl;
  EEM I_KH = Eigen::Matrix<double, EDIM, EDIM>::Identity() - (KT.transpose() * H_err);

  // update state by injecting dx
  Eigen::Matrix<double, EDIM, 1> dx(delta_x);
  dx  = (KT.transpose() * y);
  memcpy(delta_x, dx.data(), EDIM * sizeof(double));
  err_fun(in_x, delta_x, x_new);
  Eigen::Matrix<double, DIM, 1> x(x_new);

  // update cov
  P = ((I_KH * P) * I_KH.transpose()) + ((KT.transpose() * R) * KT);

  // copy out state
  memcpy(in_x, x.data(), DIM * sizeof(double));
  memcpy(in_P, P.data(), EDIM * EDIM * sizeof(double));
  memcpy(in_z, y.data(), y.rows() * sizeof(double));
}




}
extern "C" {

void car_update_25(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_25, H_25, NULL, in_z, in_R, in_ea, MAHA_THRESH_25);
}
void car_update_24(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<2, 3, 0>(in_x, in_P, h_24, H_24, NULL, in_z, in_R, in_ea, MAHA_THRESH_24);
}
void car_update_30(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_30, H_30, NULL, in_z, in_R, in_ea, MAHA_THRESH_30);
}
void car_update_26(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_26, H_26, NULL, in_z, in_R, in_ea, MAHA_THRESH_26);
}
void car_update_27(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_27, H_27, NULL, in_z, in_R, in_ea, MAHA_THRESH_27);
}
void car_update_29(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_29, H_29, NULL, in_z, in_R, in_ea, MAHA_THRESH_29);
}
void car_update_28(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_28, H_28, NULL, in_z, in_R, in_ea, MAHA_THRESH_28);
}
void car_update_31(double *in_x, double *in_P, double *in_z, double *in_R, double *in_ea) {
  update<1, 3, 0>(in_x, in_P, h_31, H_31, NULL, in_z, in_R, in_ea, MAHA_THRESH_31);
}
void car_err_fun(double *nom_x, double *delta_x, double *out_2290816945354955451) {
  err_fun(nom_x, delta_x, out_2290816945354955451);
}
void car_inv_err_fun(double *nom_x, double *true_x, double *out_8668625092093883095) {
  inv_err_fun(nom_x, true_x, out_8668625092093883095);
}
void car_H_mod_fun(double *state, double *out_4743823478057154038) {
  H_mod_fun(state, out_4743823478057154038);
}
void car_f_fun(double *state, double dt, double *out_3260309363242572689) {
  f_fun(state,  dt, out_3260309363242572689);
}
void car_F_fun(double *state, double dt, double *out_7845718988328896002) {
  F_fun(state,  dt, out_7845718988328896002);
}
void car_h_25(double *state, double *unused, double *out_1560465764258318145) {
  h_25(state, unused, out_1560465764258318145);
}
void car_H_25(double *state, double *unused, double *out_1652686704141933712) {
  H_25(state, unused, out_1652686704141933712);
}
void car_h_24(double *state, double *unused, double *out_8424072936813750229) {
  h_24(state, unused, out_8424072936813750229);
}
void car_H_24(double *state, double *unused, double *out_519962894863565854) {
  H_24(state, unused, out_519962894863565854);
}
void car_h_30(double *state, double *unused, double *out_6202799492562354296) {
  h_30(state, unused, out_6202799492562354296);
}
void car_H_30(double *state, double *unused, double *out_4171019662649182339) {
  H_30(state, unused, out_4171019662649182339);
}
void car_h_26(double *state, double *unused, double *out_196242024851494850) {
  h_26(state, unused, out_196242024851494850);
}
void car_H_26(double *state, double *unused, double *out_2088816614732122512) {
  H_26(state, unused, out_2088816614732122512);
}
void car_h_27(double *state, double *unused, double *out_8666281197547850443) {
  h_27(state, unused, out_8666281197547850443);
}
void car_H_27(double *state, double *unused, double *out_6394613733833125556) {
  H_27(state, unused, out_6394613733833125556);
}
void car_h_29(double *state, double *unused, double *out_1201126224901508651) {
  h_29(state, unused, out_1201126224901508651);
}
void car_H_29(double *state, double *unused, double *out_4681251006963574523) {
  H_29(state, unused, out_4681251006963574523);
}
void car_h_28(double *state, double *unused, double *out_6818187523072045103) {
  h_28(state, unused, out_6818187523072045103);
}
void car_H_28(double *state, double *unused, double *out_401148010105956051) {
  H_28(state, unused, out_401148010105956051);
}
void car_h_31(double *state, double *unused, double *out_5171293783944956096) {
  h_31(state, unused, out_5171293783944956096);
}
void car_H_31(double *state, double *unused, double *out_1683332666018894140) {
  H_31(state, unused, out_1683332666018894140);
}
void car_predict(double *in_x, double *in_P, double *in_Q, double dt) {
  predict(in_x, in_P, in_Q, dt);
}
void car_set_mass(double x) {
  set_mass(x);
}
void car_set_rotational_inertia(double x) {
  set_rotational_inertia(x);
}
void car_set_center_to_front(double x) {
  set_center_to_front(x);
}
void car_set_center_to_rear(double x) {
  set_center_to_rear(x);
}
void car_set_stiffness_front(double x) {
  set_stiffness_front(x);
}
void car_set_stiffness_rear(double x) {
  set_stiffness_rear(x);
}
}

const EKF car = {
  .name = "car",
  .kinds = { 25, 24, 30, 26, 27, 29, 28, 31 },
  .feature_kinds = {  },
  .f_fun = car_f_fun,
  .F_fun = car_F_fun,
  .err_fun = car_err_fun,
  .inv_err_fun = car_inv_err_fun,
  .H_mod_fun = car_H_mod_fun,
  .predict = car_predict,
  .hs = {
    { 25, car_h_25 },
    { 24, car_h_24 },
    { 30, car_h_30 },
    { 26, car_h_26 },
    { 27, car_h_27 },
    { 29, car_h_29 },
    { 28, car_h_28 },
    { 31, car_h_31 },
  },
  .Hs = {
    { 25, car_H_25 },
    { 24, car_H_24 },
    { 30, car_H_30 },
    { 26, car_H_26 },
    { 27, car_H_27 },
    { 29, car_H_29 },
    { 28, car_H_28 },
    { 31, car_H_31 },
  },
  .updates = {
    { 25, car_update_25 },
    { 24, car_update_24 },
    { 30, car_update_30 },
    { 26, car_update_26 },
    { 27, car_update_27 },
    { 29, car_update_29 },
    { 28, car_update_28 },
    { 31, car_update_31 },
  },
  .Hes = {
  },
  .sets = {
    { "mass", car_set_mass },
    { "rotational_inertia", car_set_rotational_inertia },
    { "center_to_front", car_set_center_to_front },
    { "center_to_rear", car_set_center_to_rear },
    { "stiffness_front", car_set_stiffness_front },
    { "stiffness_rear", car_set_stiffness_rear },
  },
  .extra_routines = {
  },
};

ekf_lib_init(car)
