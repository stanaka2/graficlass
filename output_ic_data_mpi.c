#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "graficlass.h"
#include "constants.h"
#include "prototype.h"

void ic4_mpi(int, float *, struct run_param *);
void pot_2LPT_mpi(float *, float *, struct run_param *);

#define FF(ix, iy, iz) f[(iz + (this_run->nz_tot + 2) * (iy + this_run->ny_tot * ix))]

void zeroset_f_data(float *f, uint64_t nmeshp2)
{
#pragma omp parallel for
  for(uint64_t ix = 0; ix < nmeshp2; ix++) {
    f[ix] = 0.0;
  }
}

void update_slab_mesh_padd(float *mesh, float *padd_lo, float *padd_hi, struct run_param *this_run)
{
  assert(this_run->ny_loc == this_run->ny_tot);
  assert(this_run->nz_loc == this_run->nz_tot);

  uint64_t npadd = this_run->npadd;
  uint64_t nx_loc, ny_loc, nz_loc;
  nx_loc = this_run->nx_loc;
  ny_loc = this_run->ny_loc;
  nz_loc = this_run->nz_loc;

  uint64_t ny2, nz2;
  ny2 = this_run->ny_loc / 2;
  nz2 = this_run->nz_loc / 2;

  uint64_t padd_count;
  float *buff_lo, *buff_hi;

  padd_count = npadd * ny_loc * (nz_loc + 2);
  buff_lo = (float *)malloc(sizeof(float) * padd_count);
  buff_hi = (float *)malloc(sizeof(float) * padd_count);

#pragma omp parallel for
  for(uint64_t i = 0; i < padd_count; i++) {
    padd_lo[i] = 0.0;
    padd_hi[i] = 0.0;
  }

#define MESH(ix, iy, iz) mesh[(iz + (nz_loc + 2) * (iy + ny_loc * ix))]
#define BUFF_LO(ix, iy, iz) buff_lo[(iz + (nz_loc + 2) * (iy + ny_loc * ix))]
#define BUFF_HI(ix, iy, iz) buff_hi[(iz + (nz_loc + 2) * (iy + ny_loc * ix))]

  if(nx_loc > 0) {
#pragma omp parallel for collapse(2)
    for(uint64_t ix = 0; ix < npadd; ix++) {
      for(uint64_t iy = 0; iy < ny_loc; iy++) {
        for(uint64_t iz = 0; iz < nz_loc + 2; iz++) {
          uint64_t ix_lo, ix_hi;
          ix_lo = ix;
          ix_hi = nx_loc - (ix + 1);

          BUFF_LO(ix, iy, iz) = MESH(ix_lo, iy, iz);
          BUFF_HI(ix, iy, iz) = MESH(ix_hi, iy, iz);
        }
      }
    }
  }

#undef MESH
#undef BUFF_LO
#undef BUFF_HI

  int padd_rank, padd_proc_size;
  int target_rankp, target_rankm;
  MPI_Win padd_winp, padd_winm;
  MPI_Info padd_info;

  MPI_Info_create(&padd_info);
  MPI_Info_set(padd_info, "no_locks", "true");

  /* create communicator and window */
  MPI_Comm_size(MPI_COMM_WORLD, &padd_proc_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &padd_rank);

  MPI_Win_create(padd_hi, sizeof(float) * padd_count, sizeof(float), padd_info, MPI_COMM_WORLD, &padd_winp);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, padd_winp);

  MPI_Win_create(padd_lo, sizeof(float) * padd_count, sizeof(float), padd_info, MPI_COMM_WORLD, &padd_winm);
  MPI_Win_fence(MPI_MODE_NOPRECEDE, padd_winm);

  /* copy padd from rank x to rank x+1 */
  target_rankp = padd_rank + 1;
  if(target_rankp == this_run->mpi_rank_edge) target_rankp = 0;

  if(target_rankp < this_run->mpi_rank_edge && padd_rank < this_run->mpi_rank_edge)
    MPI_Put(buff_hi, padd_count, MPI_FLOAT, target_rankp, 0, padd_count, MPI_FLOAT, padd_winm);

  /* copy padd from rank x to rank x-1 */
  target_rankm = padd_rank - 1;
  if(target_rankm == -1) target_rankm = this_run->mpi_rank_edge - 1;

  if(target_rankm < this_run->mpi_rank_edge && padd_rank < this_run->mpi_rank_edge)
    MPI_Put(buff_lo, padd_count, MPI_FLOAT, target_rankm, 0, padd_count, MPI_FLOAT, padd_winp);

  /* Synchronize MPI Window */
  MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED, padd_winp);
  MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED, padd_winm);

  MPI_Win_free(&padd_winp);
  MPI_Win_free(&padd_winm);
  MPI_Info_free(&padd_info);

  free(buff_lo);
  free(buff_hi);
}

void output_f_data_slab(float *f, char *out_dirname, char *output_type, struct run_param *this_run)
{
  assert(this_run->ny_loc == this_run->ny_tot);
  assert(this_run->nz_loc == this_run->nz_tot);
  assert(this_run->ny_loc_start == 0);
  assert(this_run->nz_loc_start == 0);

  static char filename[512];
  sprintf(filename, "%s/%s_%04d.dat", out_dirname, output_type, this_run->mpi_rank);

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

  int_tmp = this_run->mpi_nproc;
  fwrite(&int_tmp, sizeof(int), 1, output_fp); // number of file
  int_tmp = this_run->mpi_rank;
  fwrite(&int_tmp, sizeof(int), 1, output_fp); // id of file

  int_tmp = this_run->nx_loc;
  fwrite(&int_tmp, sizeof(int), 1, output_fp); // x-length
  int_tmp = this_run->nx_loc_start;
  fwrite(&int_tmp, sizeof(int), 1, output_fp); // x-start
  int_tmp = this_run->ny_loc;
  fwrite(&int_tmp, sizeof(int), 1, output_fp); // y-length
  int_tmp = this_run->ny_loc_start;
  fwrite(&int_tmp, sizeof(int), 1, output_fp); // y-start
  int_tmp = this_run->nz_loc;
  fwrite(&int_tmp, sizeof(int), 1, output_fp); // z-length
  int_tmp = this_run->nz_loc_start;
  fwrite(&int_tmp, sizeof(int), 1, output_fp); // z-start

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
  float float_tmp3[3];
  fwrite(&(this_run->mass_nu_num), sizeof(int), 1, output_fp);
  fwrite(this_run->mass_nu_deg, sizeof(int), MAX_NU_NUM, output_fp);

  float_tmp = this_run->nu_mass_tot;
  fwrite(&float_tmp, sizeof(float), 1, output_fp);

  for(int inu = 0; inu < MAX_NU_NUM; inu++) {
    float_tmp3[inu] = this_run->mass_nu_mass[inu];
  }
  fwrite(float_tmp3, sizeof(float), MAX_NU_NUM, output_fp);

  for(int inu = 0; inu < MAX_NU_NUM; inu++) {
    float_tmp3[inu] = this_run->mass_nu_frac[inu];
  }
  fwrite(float_tmp3, sizeof(float), MAX_NU_NUM, output_fp);

  for(uint64_t ix = 0; ix < this_run->nx_loc; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_loc; iy++) {
      uint64_t iz = 0;
      fwrite(&FF(ix, iy, iz), sizeof(float), this_run->nz_loc, output_fp);
    }
  }

  fclose(output_fp);
}

void output_cdm_data_mpi(struct run_param *this_run)
{
  int pk_type = 0;
  int irand_orig = this_run->irand;

  uint64_t nmeshp2 = this_run->fft_local_rsize;

  float *f, *phi_2;
  f = (float *)malloc(sizeof(float) * nmeshp2);
  phi_2 = (float *)malloc(sizeof(float) * nmeshp2);

  static char out_dirname[512];
  sprintf(out_dirname, "ic_cdm");
  make_directory(out_dirname);

/* 2LPT potential */
#ifdef __2LPT__
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 4;

  zeroset_f_data(f, nmeshp2);
  zeroset_f_data(phi_2, nmeshp2);

  ic4_mpi(pk_type, f, this_run);
  pot_2LPT_mpi(f, phi_2, this_run);
  output_f_data_slab(phi_2, out_dirname, "phi2", this_run);

  /* read same ramdom field for cdm_delta */
  this_run->irand = 2;
#endif

  /* density contrast */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 0;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);
  output_f_data_slab(f, out_dirname, "delta", this_run);

  /* read same ramdom field */
  this_run->irand = 2;

  /* x-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 1;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
  float *phi_padd_lo, *phi_padd_hi;
  uint64_t padd_count = this_run->npadd * this_run->ny_tot * (this_run->nz_tot + 2);
  phi_padd_lo = (float *)malloc(sizeof(float) * padd_count);
  phi_padd_hi = (float *)malloc(sizeof(float) * padd_count);

  update_slab_mesh_padd(phi_2, phi_padd_lo, phi_padd_hi, this_run);

#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
#define PADD_HI(ix, iy, iz) phi_padd_hi[((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))]
#define PADD_LO(ix, iy, iz) phi_padd_lo[((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))]
#pragma omp parallel for collapse(2)
  for(uint64_t ix = 0; ix < this_run->nx_loc; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t ixp = ix + 1;
        int64_t ixm = ix - 1;
        float phi_2_p, phi_2_m;
        int ix_padd = 0;

        if(ixp == this_run->nx_loc) {
          phi_2_p = PADD_HI(ix_padd, iy, iz);
        } else {
          int64_t imeshp = INDX(ixp, iy, iz);
          phi_2_p = phi_2[imeshp];
        }

        if(ixm == -1) {
          phi_2_m = PADD_LO(ix_padd, iy, iz);
        } else {
          int64_t imeshm = INDX(ixm, iy, iz);
          phi_2_m = phi_2[imeshm];
        }

        int64_t imesh = INDX(ix, iy, iz);
        f[imesh] += this_run->vfact_2LPT * (phi_2_p - phi_2_m) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#undef PADD_HI
#undef PADD_LO

  free(phi_padd_lo);
  free(phi_padd_hi);
#endif /* __2LPT__ */

  output_f_data_slab(f, out_dirname, "velx", this_run);

  /* y-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 2;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
#pragma omp parallel for collapse(2)
  for(uint64_t ix = 0; ix < this_run->nx_loc; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t iyp = iy + 1;
        int64_t iym = iy - 1;

        if(iyp == this_run->ny_tot) iyp = 0;
        if(iym == -1) iym = this_run->ny_tot - 1;

        int64_t imeshp = INDX(ix, iyp, iz);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ix, iym, iz);

        float phi_2_p, phi_2_m;
        phi_2_p = phi_2[imeshp];
        phi_2_m = phi_2[imeshm];

        f[imesh] += this_run->vfact_2LPT * (phi_2_p - phi_2_m) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data_slab(f, out_dirname, "vely", this_run);

  /* z-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 3;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
#pragma omp parallel for collapse(2)
  for(uint64_t ix = 0; ix < this_run->nx_loc; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t izp = iz + 1;
        int64_t izm = iz - 1;

        if(izp == this_run->nz_tot) izp = 0;
        if(izm == -1) izm = this_run->nz_tot - 1;

        int64_t imeshp = INDX(ix, iy, izp);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ix, iy, izm);

        float phi_2_p, phi_2_m;
        phi_2_p = phi_2[imeshp];
        phi_2_m = phi_2[imeshm];

        f[imesh] += this_run->vfact_2LPT * (phi_2_p - phi_2_m) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data_slab(f, out_dirname, "velz", this_run);

  free(f);
  free(phi_2);

  this_run->irand = irand_orig;
}

void output_baryonic_data_mpi(struct run_param *this_run)
{
  int pk_type = 1;
  int irand_orig = this_run->irand;

  /* read same ramdom field */
  this_run->irand = 2;

  uint64_t nmeshp2 = this_run->fft_local_rsize;

  float *f, *phi_2;
  f = (float *)malloc(sizeof(float) * nmeshp2);
  phi_2 = (float *)malloc(sizeof(float) * nmeshp2);

  static char out_dirname[512];
  sprintf(out_dirname, "ic_bar");
  make_directory(out_dirname);

  /* 2LPT potential */
#ifdef __2LPT__
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 4;

  zeroset_f_data(f, nmeshp2);
  zeroset_f_data(phi_2, nmeshp2);

  ic4_mpi(pk_type, f, this_run);
  pot_2LPT_mpi(f, phi_2, this_run);
  output_f_data_slab(phi_2, out_dirname, "phi2", this_run);
#endif

  /* density contrast */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 0;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);
  output_f_data_slab(f, out_dirname, "delta", this_run);

  /* x-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 1;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
  float *phi_padd_lo, *phi_padd_hi;
  uint64_t padd_count = this_run->npadd * this_run->ny_tot * (this_run->nz_tot + 2);
  phi_padd_lo = (float *)malloc(sizeof(float) * padd_count);
  phi_padd_hi = (float *)malloc(sizeof(float) * padd_count);

  update_slab_mesh_padd(phi_2, phi_padd_lo, phi_padd_hi, this_run);

#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
#define PADD_HI(ix, iy, iz) phi_padd_hi[((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))]
#define PADD_LO(ix, iy, iz) phi_padd_lo[((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))]
#pragma omp parallel for collapse(2)
  for(uint64_t ix = 0; ix < this_run->nx_loc; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t ixp = ix + 1;
        int64_t ixm = ix - 1;
        float phi_2_p, phi_2_m;
        int ix_padd = 0;

        if(ixp == this_run->nx_loc) {
          phi_2_p = PADD_HI(ix_padd, iy, iz);
        } else {
          int64_t imeshp = INDX(ixp, iy, iz);
          phi_2_p = phi_2[imeshp];
        }

        if(ixm == -1) {
          phi_2_m = PADD_LO(ix_padd, iy, iz);
        } else {
          int64_t imeshm = INDX(ixm, iy, iz);
          phi_2_m = phi_2[imeshm];
        }

        int64_t imesh = INDX(ix, iy, iz);
        f[imesh] += this_run->vfact_2LPT * (phi_2_p - phi_2_m) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#undef PADD_HI
#undef PADD_LO

  free(phi_padd_lo);
  free(phi_padd_hi);
#endif /* __2LPT__ */

  output_f_data_slab(f, out_dirname, "velx", this_run);

  /* y-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 2;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
#pragma omp parallel for collapse(2)
  for(uint64_t ix = 0; ix < this_run->nx_loc; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t iyp = iy + 1;
        int64_t iym = iy - 1;

        if(iyp == this_run->ny_tot) iyp = 0;
        if(iym == -1) iym = this_run->ny_tot - 1;

        int64_t imeshp = INDX(ix, iyp, iz);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ix, iym, iz);

        float phi_2_p, phi_2_m;
        phi_2_p = phi_2[imeshp];
        phi_2_m = phi_2[imeshm];

        f[imesh] += this_run->vfact_2LPT * (phi_2_p - phi_2_m) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data_slab(f, out_dirname, "vely", this_run);

  /* z-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 3;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
#pragma omp parallel for collapse(2)
  for(uint64_t ix = 0; ix < this_run->nx_loc; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t izp = iz + 1;
        int64_t izm = iz - 1;

        if(izp == this_run->nz_tot) izp = 0;
        if(izm == -1) izm = this_run->nz_tot - 1;

        int64_t imeshp = INDX(ix, iy, izp);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ix, iy, izm);

        float phi_2_p, phi_2_m;
        phi_2_p = phi_2[imeshp];
        phi_2_m = phi_2[imeshm];

        f[imesh] += this_run->vfact_2LPT * (phi_2_p - phi_2_m) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data_slab(f, out_dirname, "velz", this_run);

  free(f);
  free(phi_2);

  this_run->irand = irand_orig;
}

void output_neutrino_data_mpi(struct run_param *this_run, int inu)
{
  /* inu = 0,1,2 */
  int pk_type = 2 + inu;
  int irand_orig = this_run->irand;

  /* read same ramdom field */
  this_run->irand = 2;

  uint64_t nmeshp2 = this_run->fft_local_rsize;

  float *f, *phi_2;
  f = (float *)malloc(sizeof(float) * nmeshp2);
  phi_2 = (float *)malloc(sizeof(float) * nmeshp2);

  static char out_dirname[512];
  sprintf(out_dirname, "ic_nu%d", inu);
  make_directory(out_dirname);

/* 2LPT potential */
#ifdef __2LPT__
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 4;

  zeroset_f_data(f, nmeshp2);
  zeroset_f_data(phi_2, nmeshp2);

  ic4_mpi(pk_type, f, this_run);
  pot_2LPT_mpi(f, phi_2, this_run);
  output_f_data_slab(phi_2, out_dirname, "phi2", this_run);
#endif

  /* density contrast */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 0;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);
  output_f_data_slab(f, out_dirname, "delta", this_run);

  /* x-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 1;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
  float *phi_padd_lo, *phi_padd_hi;
  uint64_t padd_count = this_run->npadd * this_run->ny_tot * (this_run->nz_tot + 2);
  phi_padd_lo = (float *)malloc(sizeof(float) * padd_count);
  phi_padd_hi = (float *)malloc(sizeof(float) * padd_count);

  update_slab_mesh_padd(phi_2, phi_padd_lo, phi_padd_hi, this_run);

#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
#define PADD_HI(ix, iy, iz) phi_padd_hi[((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))]
#define PADD_LO(ix, iy, iz) phi_padd_lo[((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))]
#pragma omp parallel for collapse(2)
  for(uint64_t ix = 0; ix < this_run->nx_loc; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t ixp = ix + 1;
        int64_t ixm = ix - 1;
        float phi_2_p, phi_2_m;
        int ix_padd = 0;

        if(ixp == this_run->nx_loc) {
          phi_2_p = PADD_HI(ix_padd, iy, iz);
        } else {
          int64_t imeshp = INDX(ixp, iy, iz);
          phi_2_p = phi_2[imeshp];
        }

        if(ixm == -1) {
          phi_2_m = PADD_LO(ix_padd, iy, iz);
        } else {
          int64_t imeshm = INDX(ixm, iy, iz);
          phi_2_m = phi_2[imeshm];
        }

        int64_t imesh = INDX(ix, iy, iz);
        f[imesh] += this_run->vfact_2LPT * (phi_2_p - phi_2_m) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#undef PADD_HI
#undef PADD_LO

  free(phi_padd_lo);
  free(phi_padd_hi);
#endif /* __2LPT__ */

  output_f_data_slab(f, out_dirname, "velx", this_run);

  /* y-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 2;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
#pragma omp parallel for collapse(2)
  for(uint64_t ix = 0; ix < this_run->nx_loc; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t iyp = iy + 1;
        int64_t iym = iy - 1;

        if(iyp == this_run->ny_tot) iyp = 0;
        if(iym == -1) iym = this_run->ny_tot - 1;

        int64_t imeshp = INDX(ix, iyp, iz);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ix, iym, iz);

        float phi_2_p, phi_2_m;
        phi_2_p = phi_2[imeshp];
        phi_2_m = phi_2[imeshm];

        f[imesh] += this_run->vfact_2LPT * (phi_2_p - phi_2_m) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data_slab(f, out_dirname, "vely", this_run);

  /* z-velocity */
  this_run->xoff = 0.0;
  this_run->itide = 0;
  this_run->idim = 3;

  zeroset_f_data(f, nmeshp2);
  ic4_mpi(pk_type, f, this_run);

  /* Convert displacement to proper peculiar velocity in km/s. */
#pragma omp parallel for
  for(uint64_t i = 0; i < nmeshp2; i++) {
    f[i] *= this_run->vfact;
  }

#ifdef __2LPT__
#define INDX(ix, iy, iz) ((iz) + (this_run->nz_tot + 2) * ((iy) + this_run->ny_tot * (ix)))
#pragma omp parallel for collapse(2)
  for(uint64_t ix = 0; ix < this_run->nx_loc; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      for(uint64_t iz = 0; iz < this_run->nz_tot; iz++) {

        int64_t izp = iz + 1;
        int64_t izm = iz - 1;

        if(izp == this_run->nz_tot) izp = 0;
        if(izm == -1) izm = this_run->nz_tot - 1;

        int64_t imeshp = INDX(ix, iy, izp);
        int64_t imesh = INDX(ix, iy, iz);
        int64_t imeshm = INDX(ix, iy, izm);

        float phi_2_p, phi_2_m;
        phi_2_p = phi_2[imeshp];
        phi_2_m = phi_2[imeshm];

        f[imesh] += this_run->vfact_2LPT * (phi_2_p - phi_2_m) / (2.0 * this_run->dx);
      }
    }
  }
#undef INDX
#endif /* __2LPT__ */

  output_f_data_slab(f, out_dirname, "velz", this_run);

  free(f);
  free(phi_2);

  this_run->irand = irand_orig;
}
