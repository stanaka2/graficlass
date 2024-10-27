#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "graficlass.h"
#include "constants.h"

void ic4(int, float *, struct run_param *);
void pot_2LPT(float *, float *, struct run_param *);

#define FF(ix, iy, iz) f[(iz + (this_run->nz_tot + 2) * (iy + this_run->ny_tot * ix))]

void zeroset_f_data(float *f, uint64_t nmeshp2)
{
#pragma omp parallel for
  for(uint64_t ix = 0; ix < nmeshp2; ix++) {
    f[ix] = 0.0;
  }
}

void output_f_data(float *f, char *filename, struct run_param *this_run)
{
  FILE *output_fp;
  output_fp = fopen(filename, "w");

  if(output_fp == NULL) {
    fprintf(stderr, "Cannot create %s file.\n", filename);
    exit(EXIT_FAILURE);
  }

  int int_tmp;
  float float_tmp;
  double double_tmp;

  int_tmp = this_run->nx_tot;
  fwrite(&int_tmp, sizeof(int), 1, output_fp);
  int_tmp = this_run->ny_tot;
  fwrite(&int_tmp, sizeof(int), 1, output_fp);
  int_tmp = this_run->nz_tot;
  fwrite(&int_tmp, sizeof(int), 1, output_fp);

  double_tmp = this_run->nx_tot * this_run->dx;
  fwrite(&double_tmp, sizeof(double), 1, output_fp);

  float_tmp = this_run->astart;
  fwrite(&float_tmp, sizeof(float), 1, output_fp);
  float_tmp = this_run->cosm.omegam;
  fwrite(&float_tmp, sizeof(float), 1, output_fp);
  float_tmp = this_run->cosm.omegav;
  fwrite(&float_tmp, sizeof(float), 1, output_fp);
  float_tmp = this_run->cosm.omegab;
  fwrite(&float_tmp, sizeof(float), 1, output_fp);
  float_tmp = this_run->cosm.h0 / 100.0;
  fwrite(&float_tmp, sizeof(float), 1, output_fp);

  /* add Omega ratiaton */
  float_tmp = this_run->cosm.omegar;
  fwrite(&float_tmp, sizeof(float), 1, output_fp);

  /* add neutrinos information */
  fwrite(&(this_run->mass_nu_num), sizeof(int), 1, output_fp);
  fwrite(this_run->mass_nu_deg, sizeof(int), MAX_NU_NUM, output_fp);

  float_tmp = this_run->nu_mass_tot;
  fwrite(&float_tmp, sizeof(float), 1, output_fp);

  float float_tmp3[3];
  for(int inu = 0; inu < MAX_NU_NUM; inu++) {
    float_tmp3[inu] = this_run->mass_nu_mass[inu];
  }
  fwrite(float_tmp3, sizeof(float), MAX_NU_NUM, output_fp);

  for(int inu = 0; inu < MAX_NU_NUM; inu++) {
    float_tmp3[inu] = this_run->mass_nu_frac[inu];
  }
  fwrite(float_tmp3, sizeof(float), MAX_NU_NUM, output_fp);

  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      uint64_t iz = 0;
      fwrite(&FF(ix, iy, iz), sizeof(float), this_run->nz_tot, output_fp);
    }
  }

  fclose(output_fp);
}

void output_cdm_data(struct run_param *this_run)
{
  int pk_type = 0;
  int irand_orig = this_run->irand;

  uint64_t nmeshp2 = this_run->nx_tot * this_run->ny_tot * (this_run->nz_tot + 2);

  float *f, *phi_2;
  f = (float *)malloc(sizeof(float) * nmeshp2);
  phi_2 = (float *)malloc(sizeof(float) * nmeshp2);

  char *delta_filename = "ic_cdm_delta.dat";
  char *velx_filename = "ic_cdm_velx.dat";
  char *vely_filename = "ic_cdm_vely.dat";
  char *velz_filename = "ic_cdm_velz.dat";
  char *phi2_filename = "ic_cdm_phi2.dat";

  /* 2LPT potential */
#ifdef __2LPT__
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 4;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);
  pot_2LPT(f, phi_2, this_run);
  output_f_data(phi_2, phi2_filename, this_run);

  /* read same ramdom field for cdm_delta */
  this_run->irand = 2;
#endif

  /* density contrast */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 0;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);
  output_f_data(f, delta_filename, this_run);

  /* read same ramdom field */
  this_run->irand = 2;

  /* x-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 1;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
#pragma omp parallel for
  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t ixp = ix + 1;
        if(ixp == this_run->nx_tot) ixp = 0;
        int64_t ixm = ix - 1;
        if(ixm == -1) ixm = this_run->nx_tot - 1;

        int64_t imeshp = INDX(ixp, iy, iz);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ixm, iy, iz);

        f[imesh] += this_run->vfact_2LPT * (phi_2[imeshp] - phi_2[imeshm]) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data(f, velx_filename, this_run);

  /* y-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 2;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
#pragma omp parallel for
  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t iyp = iy + 1;
        if(iyp == this_run->ny_tot) iyp = 0;
        int64_t iym = iy - 1;
        if(iym == -1) iym = this_run->ny_tot - 1;

        int64_t imeshp = INDX(ix, iyp, iz);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ix, iym, iz);

        f[imesh] += this_run->vfact_2LPT * (phi_2[imeshp] - phi_2[imeshm]) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data(f, vely_filename, this_run);

  /* z-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 3;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#pragma omp parallel for
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t izp = iz + 1;
        if(izp == this_run->nz_tot) izp = 0;
        int64_t izm = iz - 1;
        if(izm == -1) izm = this_run->nz_tot - 1;

        uint64_t imeshp = INDX(ix, iy, izp);
        uint64_t imesh = INDX(ix, iy, iz);
        uint64_t imeshm = INDX(ix, iy, izm);

        f[imesh] += this_run->vfact_2LPT * (phi_2[imeshp] - phi_2[imeshm]) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data(f, velz_filename, this_run);

  free(f);
  free(phi_2);

  this_run->irand = irand_orig;
}

void output_baryonic_data(struct run_param *this_run)
{
  int pk_type = 1;
  int irand_orig = this_run->irand;

  /* read same ramdom field */
  this_run->irand = 2;

  uint64_t nmeshp2 = this_run->nx_tot * this_run->ny_tot * (this_run->nz_tot + 2);

  float *f, *phi_2;
  f = (float *)malloc(sizeof(float) * nmeshp2);
  phi_2 = (float *)malloc(sizeof(float) * nmeshp2);

  char *delta_filename = "ic_bar_delta.dat";
  char *velx_filename = "ic_bar_velx.dat";
  char *vely_filename = "ic_bar_vely.dat";
  char *velz_filename = "ic_bar_velz.dat";
  char *phi2_filename = "ic_bar_phi2.dat";

/* 2LPT potential */
#ifdef __2LPT__
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 4;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);
  pot_2LPT(f, phi_2, this_run);
  output_f_data(phi_2, phi2_filename, this_run);
#endif

  /* density contrast */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 0;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);
  output_f_data(f, delta_filename, this_run);

  /* x-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 1;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#pragma omp parallel for
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    int64_t ixp = ix + 1;
    if(ixp == this_run->nx_tot) ixp = 0;
    int64_t ixm = ix - 1;
    if(ixm == -1) ixm = this_run->nx_tot - 1;

    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {
        int64_t imeshp = INDX(ixp, iy, iz);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ixm, iy, iz);

        f[imesh] += this_run->vfact_2LPT * (phi_2[imeshp] - phi_2[imeshm]) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data(f, velx_filename, this_run);

  /* y-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 2;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#pragma omp parallel for
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      int64_t iyp = iy + 1;
      if(iyp == this_run->ny_tot) iyp = 0;
      int64_t iym = iy - 1;
      if(iym == -1) iym = this_run->ny_tot - 1;
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {
        int64_t imeshp = INDX(ix, iyp, iz);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ix, iym, iz);

        f[imesh] += this_run->vfact_2LPT * (phi_2[imeshp] - phi_2[imeshm]) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data(f, vely_filename, this_run);

  /* z-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 3;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#pragma omp parallel for
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {
        int64_t izp = iz + 1;
        if(izp == this_run->nz_tot) izp = 0;
        int64_t izm = iz - 1;
        if(izm == -1) izm = this_run->nz_tot - 1;

        uint64_t imeshp = INDX(ix, iy, izp);
        uint64_t imesh = INDX(ix, iy, iz);
        uint64_t imeshm = INDX(ix, iy, izm);

        f[imesh] += this_run->vfact_2LPT * (phi_2[imeshp] - phi_2[imeshm]) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data(f, velz_filename, this_run);

  free(f);
  free(phi_2);

  this_run->irand = irand_orig;
}

void output_neutrino_data(struct run_param *this_run, int inu)
{
  /* inu = 0,1,2 */
  int pk_type = 2 + inu;
  int irand_orig = this_run->irand;

  /* read same ramdom field */
  this_run->irand = 2;

  uint64_t nmeshp2 = this_run->nx_tot * this_run->ny_tot * (this_run->nz_tot + 2);

  float *f, *phi_2;
  f = (float *)malloc(sizeof(float) * nmeshp2);
  phi_2 = (float *)malloc(sizeof(float) * nmeshp2);

  char delta_filename[256], velx_filename[256], vely_filename[256], velz_filename[256], phi2_filename[256];

  sprintf(delta_filename, "ic_nu%d_delta.dat", inu);
  sprintf(velx_filename, "ic_nu%d_velx.dat", inu);
  sprintf(vely_filename, "ic_nu%d_vely.dat", inu);
  sprintf(velz_filename, "ic_nu%d_velz.dat", inu);
  sprintf(phi2_filename, "ic_nu%d_phi2.dat", inu);

/* 2LPT potential */
#ifdef __2LPT__
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 4;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);
  pot_2LPT(f, phi_2, this_run);
  output_f_data(phi_2, phi2_filename, this_run);
#endif

  /* density contrast */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 0;

  ic4(pk_type, f, this_run);
  output_f_data(f, delta_filename, this_run);

  /* x-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 1;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#pragma omp parallel for
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    int64_t ixp = ix + 1;
    if(ixp == this_run->nx_tot) ixp = 0;
    int64_t ixm = ix - 1;
    if(ixm == -1) ixm = this_run->nx_tot - 1;

    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {
        int64_t imeshp = INDX(ixp, iy, iz);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ixm, iy, iz);

        f[imesh] += this_run->vfact_2LPT * (phi_2[imeshp] - phi_2[imeshm]) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data(f, velx_filename, this_run);

  /* y-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 2;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#pragma omp parallel for
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      int64_t iyp = iy + 1;
      if(iyp == this_run->ny_tot) iyp = 0;
      int64_t iym = iy - 1;
      if(iym == -1) iym = this_run->ny_tot - 1;
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {
        int64_t imeshp = INDX(ix, iyp, iz);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ix, iym, iz);

        f[imesh] += this_run->vfact_2LPT * (phi_2[imeshp] - phi_2[imeshm]) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data(f, vely_filename, this_run);

  /* z-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 3;

  zeroset_f_data(f, nmeshp2);
  ic4(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#pragma omp parallel for
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {
        int64_t izp = iz + 1;
        if(izp == this_run->nz_tot) izp = 0;
        int64_t izm = iz - 1;
        if(izm == -1) izm = this_run->nz_tot - 1;

        uint64_t imeshp = INDX(ix, iy, izp);
        uint64_t imesh = INDX(ix, iy, iz);
        uint64_t imeshm = INDX(ix, iy, izm);

        f[imesh] += this_run->vfact_2LPT * (phi_2[imeshp] - phi_2[imeshm]) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data(f, velz_filename, this_run);

  free(f);
  free(phi_2);

  this_run->irand = irand_orig;
}
