#pragma once

#include "graficlass.h"
#include "fftw_param.h"

/* time_funcs.c */
double calc_aexp(double, struct cosmology *);
double calc_tau(double, struct cosmology *);
double calc_dtauda(double, struct cosmology *);
double calc_dladt(double, struct cosmology *);
double calc_dplus(double, struct cosmology *);
double calc_ddplus(double, struct cosmology *);
double calc_adp(double, struct cosmology *);
double calc_fomega(double, struct cosmology *);
double calc_fomega_2LPT(double, struct cosmology *);

/* power_funcs.c */
double calc_dsigma(double, struct cosmology *);
void pini(struct run_param *);
double calc_dc2(double, struct cosmology *);
double calc_dc2l(double, struct cosmology *);
double calc_dphid(double, struct cosmology *);
double calc_dtconda(double, struct cosmology *);
double calc_tcon(double, struct cosmology *);
double calc_dsig8(double, struct cosmology *);
double calc_uj2(double, double, double);
double calc_aj2(double);
double calc_bj2(double, double);
double calc_rombin(double (*func)(double, struct cosmology *), double, double, double, struct cosmology *);
double calc_p(double, double, struct cosmology *);
double calc_pk_tot(double, double, struct cosmology *);
double calc_pk_cb(double, double, struct cosmology *);
double calc_pk_cdm(double, double, struct cosmology *);
double calc_pk_bar(double, double, struct cosmology *);
double calc_pk_nu(double, double, struct cosmology *, int);

/* transfer_table.c */
double calc_ttot_class(double, double, struct cosmology *);
double calc_tcb_class(double, double, struct cosmology *);
double calc_tcdm_class(double, double, struct cosmology *);
double calc_tbar_class(double, double, struct cosmology *);
double calc_tnu1_class(double, double, struct cosmology *);
double calc_tnu2_class(double, double, struct cosmology *);
double calc_tnu3_class(double, double, struct cosmology *);

/* util_funcs.c */
void fit_splder(double *, double *, double *, int);
void splini(double *, int);

/* output_ic_data.c */
void output_cb_data(struct run_param *);
void output_cdm_data(struct run_param *);
void output_baryonic_data(struct run_param *);
void output_neutrino_data(struct run_param *, int);

/* output_ic_data_mpi.c */
void output_cdm_data_mpi(struct run_param *);
void output_baryonic_data_mpi(struct run_param *);
void output_neutrino_data_mpi(struct run_param *, int);

/* system_call */
void make_directory(char *);
void remove_file(char *);
void remove_dir(char *);
