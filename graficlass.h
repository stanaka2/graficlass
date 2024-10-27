#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>

#define NP (1024)
#define NPX (NP)
#define NPY (NP)
#define NPZ (NP)

#define NKMAX (2001)
#define NT (11)

#define MAX_NU_NUM (3)

struct cosmology {
  double omegam, omegab, omegav, omegak, h0, omegar;

  double sigma;
  double an, anorml, pnorm;
  double asig, dx, ak, dk;
  double a00, a10, scale;
  double akmin, akmax;
  double akminl, akmaxl;
  int icase;
  int mass_nu_num;

  double deltat2[NKMAX], ddeltat2[NKMAX];
  double tmat[NT][NKMAX], dtmat[NT][NKMAX];
  double tcb[NT][NKMAX], dtcb[NT][NKMAX];
  double tcdm[NT][NKMAX], dtcdm[NT][NKMAX];
  double tbar[NT][NKMAX], dtbar[NT][NKMAX];

  char *tkfilename, *pkfilename;
};

struct run_param {
  int mpi_nproc, mpi_rank;
  int mpi_rank_edge;

  uint64_t nx_tot, ny_tot, nz_tot;
  uint64_t np_tot;

  uint64_t nx_loc, ny_loc, nz_loc;
  uint64_t nx_loc_start, ny_loc_start, nz_loc_start;
  uint64_t np_loc;
  uint64_t fft_local_rsize;
  int howmany, fft_rblock, fft_cblock;
  int npadd;

  uint64_t m1s, m2s, m3s;
  uint64_t m1offt, m2offt, m3offt;
  double x1o, x2o, x3o;

  double dx, dxr;
  double ak1, ak2; // k-min,k-max

  struct cosmology cosm;

  double tcmb;
  double astart;
  double init_zred;

  double nu_mass_tot;
  int mass_nu_num;
  int mass_nu_deg[MAX_NU_NUM];
  double mass_nu_mass[MAX_NU_NUM];
  double mass_nu_frac[MAX_NU_NUM];

  double vfact, vfact_2LPT;
  double sigma, sigstart;

  double nrefine;
  int iwarn1, iwarn2;
  int icase;
  int hanning;
  int irand, iseed;
  int itide, idim;
  double xoff;

  /* a table */
  int ntab;
  double atab[NT];

  char *tkfilename, *pkfilename;
  char *noisefilename;

  int fft_thread_flag;
};

#define SQR(x) ((x) * (x))
#define CUBE(x) ((x) * (x) * (x))
