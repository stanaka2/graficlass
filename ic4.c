#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <omp.h>
#include <fftw3.h>
#include "MT.h"

#include "graficlass.h"
#include "constants.h"
#include "prototype.h"

/* on  : Discrete Green's functions */
/* off : Continuous Green's functions (Poor Man’s Poisson Solver) */
#define __USE_GREEN_FUNC__

#define FF(ix, iy, iz) f[(iz + (nz + 2) * (iy + ny * ix))]
#define FC(ix, iy, iz) f_hat[(iz + (nz2 + 1) * (iy + ny * ix))]

#define cmplx_re(c) ((c)[0])
#define cmplx_im(c) ((c)[1])

struct complexd {
  double real;
  double imag;
};

struct complexf {
  float real;
  float imag;
};

struct complexd cmplx_add(struct complexd a, struct complexd b)
{
  struct complexd z;
  z.real = a.real + b.real;
  z.imag = a.imag + b.imag;
  return z;
}

struct complexf cmplxf_add(struct complexf a, struct complexf b)
{
  struct complexf z;
  z.real = a.real + b.real;
  z.imag = a.imag + b.imag;
  return z;
}

struct complexd cmplx_mul(struct complexd a, struct complexd b)
{
  struct complexd z;
  z.real = a.real * b.real - a.imag * b.imag;
  z.imag = a.real * b.imag + a.imag * b.real;
  return z;
}

struct complexf cmplxf_mul(struct complexf a, struct complexf b)
{
  struct complexf z;
  z.real = a.real * b.real - a.imag * b.imag;
  z.imag = a.real * b.imag + a.imag * b.real;
  return z;
}

double cmplx_abs2(struct complexd a)
{
  double z;
  z = a.real * a.real + a.imag * a.imag;
  return z;
}

float cmplxf_abs2(struct complexf a)
{
  float z;
  z = a.real * a.real + a.imag * a.imag;
  return z;
}

/* Box-Muller Method */
double normalrand()
{
  /* rand : (0, 1) */
  double xr = sqrt(-2.0 * log(genrand_real2()));
  double z1 = xr * sin(TWO_PI * genrand_real2());
  // double z2=xr*cos(TWO_PI*genrand_real2());

  // sigma=1 , ave=0;
  // return z1*sigma + ave;
  return z1;
}

/* for 64-bit */
double normalrand64()
{
  /* rand : (0, 1) */
  double xr = sqrt(-2.0 * log(genrand64_real2()));
  double z1 = xr * sin(TWO_PI * genrand64_real2());
  return z1;
}

#define INDX(ix, iy, iz) ((iz) + (nz + 2) * ((iy) + ny * (ix)))
#define KINDX(ix, iy, iz) ((iz) + (nz2 + 1) * ((iy) + ny * (ix)))
#define GKINDX(ix, iy, iz) ((iz) + (nz2 + 1) * ((iy) + (ny2 + 1) * (ix)))

void setup_GK(double *gk, struct run_param *this_run)
{
  uint64_t nx, ny, nz;
  nx = this_run->nx_tot;
  ny = this_run->ny_tot;
  nz = this_run->nz_tot;

  assert(nx == ny);
  assert(ny == nz);

  uint64_t ng = nx / 2 + 1;

  double *sins = (double *)malloc(sizeof(double) * ng);

  for(int i = 0; i < ng; i++) {
    sins[i] = SQR(sin(PI * (double)i / nx));
  }

  //  double pins = -PI/SQR(this_run->dx);
  double pins = -0.25 * SQR(this_run->dx);

#define GK(ix, iy, iz) (gk[iz + ng * (iy + ng * ix)])

#pragma omp parallel for collapse(2)
  for(int ix = 1; ix < ng; ix++) {
    for(int iy = 0; iy < ng; iy++) {
      for(int iz = 0; iz < ng; iz++) {
        GK(ix, iy, iz) = pins / (sins[ix] + sins[iy] + sins[iz]);
      }
    }
  }

#pragma omp parallel for collapse(2)
  for(int iy = 1; iy < ng; iy++) {
    for(int iz = 0; iz < ng; iz++) {
      GK(0, iy, iz) = pins / (sins[0] + sins[iy] + sins[iz]);
    }
  }

#pragma omp parallel for
  for(int iz = 1; iz < ng; iz++) {
    GK(0, 0, iz) = pins / (sins[0] + sins[0] + sins[iz]);
  }

  GK(0, 0, 0) = 0.0;
#undef GK

  free(sins);
}

void pot_2LPT(float *phi, float *phi_2, struct run_param *this_run)
{
  // Compute the 2LPT potential
  // the input variable phi is the solution of the 1-st order Poisson equation
  // (D13a) of Scoccimarro 1998.

  fftwf_plan forward_planf, backward_planf;

  fftwf_init_threads();
  fftwf_plan_with_nthreads(omp_get_max_threads());

  uint64_t nx = this_run->nx_tot;
  uint64_t ny = this_run->ny_tot;
  uint64_t nz = this_run->nz_tot;

  uint64_t nx2 = this_run->nx_tot / 2;
  uint64_t ny2 = this_run->ny_tot / 2;
  uint64_t nz2 = this_run->nz_tot / 2;

  double dx2 = SQR(this_run->dx);
  double dkx = TWO_PI / (nx * this_run->dx);
  double dky = TWO_PI / (ny * this_run->dx);
  double dkz = TWO_PI / (nz * this_run->dx);
  //  float akmax=TWO_PI/this_run->dx;

  float *phi_xx, *phi_yy, *phi_zz;

  uint64_t nmeshp2 = nx * ny * (nz + 2);

  phi_xx = (float *)malloc(sizeof(float) * nmeshp2);
  phi_yy = (float *)malloc(sizeof(float) * nmeshp2);
  phi_zz = (float *)malloc(sizeof(float) * nmeshp2);

  // Initially compute the source term of the Poisson equation
  // (D13b) of Scoccimarro 1998.
  // \sum_{i>j} phi1_ii*phi1_jj-[phi1_ij]^2

#pragma omp parallel for collapse(2)
  for(uint64_t ix = 0; ix < nx; ix++) {
    for(uint64_t iy = 0; iy < ny; iy++) {
      for(uint64_t iz = 0; iz < nz; iz++) {

        int64_t ixp = ix + 1;
        if(ixp == nx) ixp = 0;
        int64_t ixm = ix - 1;
        if(ixm == -1) ixm = nx - 1;

        uint64_t imeshp = INDX(ixp, iy, iz);
        uint64_t imesh = INDX(ix, iy, iz);
        uint64_t imeshm = INDX(ixm, iy, iz);

        phi_xx[imesh] = (phi[imeshp] - 2.0 * phi[imesh] + phi[imeshm]) / dx2;
      }
    }
  }

#pragma omp parallel for collapse(2)
  for(uint64_t ix = 0; ix < nx; ix++) {
    for(uint64_t iy = 0; iy < ny; iy++) {
      for(uint64_t iz = 0; iz < nz; iz++) {

        int64_t iyp = iy + 1;
        if(iyp == ny) iyp = 0;
        int64_t iym = iy - 1;
        if(iym == -1) iym = ny - 1;

        uint64_t imeshp = INDX(ix, iyp, iz);
        uint64_t imesh = INDX(ix, iy, iz);
        uint64_t imeshm = INDX(ix, iym, iz);

        phi_yy[imesh] = (phi[imeshp] - 2.0 * phi[imesh] + phi[imeshm]) / dx2;
      }
    }
  }

#pragma omp parallel for
  for(uint64_t ix = 0; ix < nx; ix++) {
    for(uint64_t iy = 0; iy < ny; iy++) {
      for(uint64_t iz = 0; iz < nz; iz++) {
        int64_t izp = iz + 1;
        if(izp == nz) izp = 0;
        int64_t izm = iz - 1;
        if(izm == -1) izm = nz - 1;

        uint64_t imeshp = INDX(ix, iy, izp);
        uint64_t imesh = INDX(ix, iy, iz);
        uint64_t imeshm = INDX(ix, iy, izm);

        phi_zz[imesh] = (phi[imeshp] - 2.0 * phi[imesh] + phi[imeshm]) / dx2;
      }
    }
  }

#pragma omp parallel for
  for(uint64_t ix = 0; ix < nx; ix++) {
    for(uint64_t iy = 0; iy < ny; iy++) {
      for(int64_t iz = 0; iz < nz; iz++) {
        uint64_t imesh = INDX(ix, iy, iz);

        phi_2[imesh] = phi_xx[imesh] * phi_yy[imesh];
        phi_2[imesh] += phi_yy[imesh] * phi_zz[imesh];
        phi_2[imesh] += phi_xx[imesh] * phi_zz[imesh];
      }
    }
  }

  // derivative wrt xy
#pragma omp parallel for
  for(uint64_t ix = 0; ix < nx; ix++) {
    int64_t ixp = ix + 1;
    if(ixp == nx) ixp = 0;
    int64_t ixm = ix - 1;
    if(ixm == -1) ixm = nx - 1;
    for(uint64_t iy = 0; iy < ny; iy++) {
      int64_t iyp = iy + 1;
      if(iyp == ny) iyp = 0;
      int64_t iym = iy - 1;
      if(iym == -1) iym = ny - 1;
      for(int64_t iz = 0; iz < nz; iz++) {
        uint64_t imesh = INDX(ix, iy, iz);
        uint64_t imeshpp = INDX(ixp, iyp, iz);
        uint64_t imeshmp = INDX(ixp, iym, iz);
        uint64_t imeshpm = INDX(ixm, iyp, iz);
        uint64_t imeshmm = INDX(ixm, iym, iz);

        float phi_xy = (phi[imeshpp] - phi[imeshpm] - phi[imeshmp] + phi[imeshmm]) / (4.0 * dx2);

        phi_2[imesh] -= SQR(phi_xy);
      }
    }
  }

  // derivative wrt yz
#pragma omp parallel for
  for(uint64_t ix = 0; ix < nx; ix++) {
    for(uint64_t iy = 0; iy < ny; iy++) {
      int64_t iyp = iy + 1;
      if(iyp == ny) iyp = 0;
      int64_t iym = iy - 1;
      if(iym == -1) iym = ny - 1;
      for(int64_t iz = 0; iz < nz; iz++) {
        int64_t izp = iz + 1;
        if(izp == nz) izp = 0;
        int64_t izm = iz - 1;
        if(izm == -1) izm = nz - 1;

        uint64_t imesh = INDX(ix, iy, iz);
        uint64_t imeshpp = INDX(ix, iyp, izp);
        uint64_t imeshmp = INDX(ix, iym, izp);
        uint64_t imeshpm = INDX(ix, iyp, izm);
        uint64_t imeshmm = INDX(ix, iym, izm);

        float phi_yz = (phi[imeshpp] - phi[imeshpm] - phi[imeshmp] + phi[imeshmm]) / (4.0 * dx2);

        phi_2[imesh] -= SQR(phi_yz);
      }
    }
  }

  // derivative wrt zx
#pragma omp parallel for
  for(uint64_t ix = 0; ix < nx; ix++) {
    int64_t ixp = ix + 1;
    if(ixp == nx) ixp = 0;
    int64_t ixm = ix - 1;
    if(ixm == -1) ixm = nx - 1;
    for(uint64_t iy = 0; iy < ny; iy++) {
      for(int64_t iz = 0; iz < nz; iz++) {
        int64_t izp = iz + 1;
        if(izp == nz) izp = 0;
        int64_t izm = iz - 1;
        if(izm == -1) izm = nz - 1;

        uint64_t imesh = INDX(ix, iy, iz);
        uint64_t imeshpp = INDX(ixp, iy, izp);
        uint64_t imeshmp = INDX(ixp, iy, izm);
        uint64_t imeshpm = INDX(ixm, iy, izp);
        uint64_t imeshmm = INDX(ixm, iy, izm);

        float phi_zx = (phi[imeshpp] - phi[imeshpm] - phi[imeshmp] + phi[imeshmm]) / (4.0 * dx2);

        phi_2[imesh] -= SQR(phi_zx);
      }
    }
  }

#ifdef __DEBUG__
  FILE *fp1 = fopen("phi_1.dat", "w");
  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      int iz = 20;
      fprintf(fp1, "%ld %ld %ld %14.6e\n", ix, iy, iz,
              phi_2[iz + (this_run->nz_tot + 2) * (iy + this_run->ny_tot * ix)]);
    }
    fprintf(fp1, "\n");
  }
  fclose(fp1);
#endif

  fftwf_complex *phi_2_hat;
  phi_2_hat = (fftwf_complex *)phi_2;
  /* Transform phi_2 to k-space. */
  forward_planf = fftwf_plan_dft_r2c_3d(nx, ny, nz, (float *)phi_2, (fftwf_complex *)phi_2_hat, FFTW_ESTIMATE);

  fftwf_execute(forward_planf);
  fftwf_destroy_plan(forward_planf);

  printf("Done forward FFT in pot_2LPT.\n");
  fflush(stdout);

  double *gk = (double *)malloc(sizeof(double) * (nx2 + 1) * (ny2 + 1) * (nz2 + 1));
  setup_GK(gk, this_run);

  for(int ix = 0; ix < nx2; ix++) {
    int ixp = nx2 + ix;
    int ixm = nx2 - ix;
    for(int iy = 0; iy < ny2; iy++) {
      int iyp = ny2 + iy;
      int iym = ny2 - iy;
      for(int iz = 0; iz < nz2 + 1; iz++) {
        cmplx_re(phi_2_hat[KINDX(ix, iy, iz)]) *= gk[GKINDX(ix, iy, iz)];
        cmplx_re(phi_2_hat[KINDX(ixp, iy, iz)]) *= gk[GKINDX(ixm, iy, iz)];
        cmplx_re(phi_2_hat[KINDX(ix, iyp, iz)]) *= gk[GKINDX(ix, iym, iz)];
        cmplx_re(phi_2_hat[KINDX(ixp, iyp, iz)]) *= gk[GKINDX(ixm, iym, iz)];

        cmplx_im(phi_2_hat[KINDX(ix, iy, iz)]) *= gk[GKINDX(ix, iy, iz)];
        cmplx_im(phi_2_hat[KINDX(ixp, iy, iz)]) *= gk[GKINDX(ixm, iy, iz)];
        cmplx_im(phi_2_hat[KINDX(ix, iyp, iz)]) *= gk[GKINDX(ix, iym, iz)];
        cmplx_im(phi_2_hat[KINDX(ixp, iyp, iz)]) *= gk[GKINDX(ixm, iym, iz)];
      }
    }
  }

  free(gk);

  /* Transform to real space. */
  backward_planf = fftwf_plan_dft_c2r_3d(nx, ny, nz, (fftwf_complex *)phi_2_hat, (float *)phi_2, FFTW_ESTIMATE);

  fftwf_execute(backward_planf);
  fftwf_destroy_plan(backward_planf);

  printf("Done barckward FFT in pot_2LPT.\n");
  fflush(stdout);

// fix the normalization
#pragma omp parallel for
  for(uint64_t imesh = 0; imesh < nx * ny * (nz + 2); imesh++) {
    phi_2[imesh] /= (double)(nx * ny * nz);
  }

#ifdef __DEBUG__
  FILE *fp2 = fopen("phi_2.dat", "w");
  for(uint64_t ix = 0; ix < this_run->nx_tot; ix++) {
    int64_t ixp = ix + 1;
    if(ixp == this_run->nx_tot) ixp = 0;
    int64_t ixm = ix - 1;
    if(ixm == -1) ixm = this_run->nx_tot - 1;
    for(uint64_t iy = 0; iy < this_run->ny_tot; iy++) {
      int64_t iyp = iy + 1;
      if(iyp == this_run->ny_tot) iyp = 0;
      int64_t iym = iy - 1;
      if(iym == -1) iym = this_run->ny_tot - 1;

      int64_t iz = 20;
      int64_t izp = iz + 1;
      if(izp == this_run->nz_tot) izp = 0;
      int64_t izm = iz - 1;
      if(izm == -1) izm = this_run->nz_tot - 1;

      int64_t imeshp, imeshm, imesh;

      imeshp = INDX(ixp, iy, iz);
      imesh = INDX(ix, iy, iz);
      imeshm = INDX(ixm, iy, iz);

      double phi_xx = phi_2[imeshp] - 2.0 * phi_2[imesh] + phi_2[imeshm];

      imeshp = INDX(ix, iyp, iz);
      imesh = INDX(ix, iy, iz);
      imeshm = INDX(ix, iym, iz);

      double phi_yy = phi_2[imeshp] - 2.0 * phi_2[imesh] + phi_2[imeshm];

      imeshp = INDX(ix, iy, izp);
      imesh = INDX(ix, iy, iz);
      imeshm = INDX(ix, iy, izm);

      double phi_zz = phi_2[imeshp] - 2.0 * phi_2[imesh] + phi_2[imeshm];

      double phi_source = (phi_xx + phi_yy + phi_zz) / SQR(this_run->dx);

      fprintf(fp2, "%ld %ld %ld %14.6e\n", ix, iy, iz, phi_source);
    }
    fprintf(fp2, "\n");
  }
  fclose(fp2);
#endif

  fftwf_cleanup_threads();
}

/*
  Generate an unconstrained sample of (rho,psi1,psi2,psi3,phi) for
  idim=0,1,2,3,4.
  Input: idim, irand, iseed, itide, m?s, m?off, hanning, filename,
  astart, pk, dx, xoff
  irand=0: use randg to generate white noise, don't save.
  irand=1: use randg to generate white noise, then save in real space
  in filename.
  irand=2: read filename to get random numbers.
  iseed: 9-digit integer random number seed.  Beware that rand8 does not
  give the same random numbers on 32-bit and 64-bit machines!
  itide=0 to use full subvolume for computing f.
  itide=1 to set xi=0 inside subvolume so as to get outer field.
  itide=-1 to set xi=0 outside subvolume so as to get inner field.
  m?s = size of next-level subvolume to split if itide.ne.0.
  m?off = offset of next-level subvolume to split if itide.ne.0
  hanning=T to apply hanning filter to f.
  hanning=F to not apply hanning filter to f.
  filename = file containing random numbers in real space.
  astart = expansion factor
  pk(ak,astart) = power spectrum function for wavenumber ak
  dx = grid spacing.
  xoff = offset to evaluate fields (e.g. use to shift baryon or cdm fields).
  Output: f=fc (sampled field in real space), fm (maximum absolute value of f).
  N.B. f and fc must point to the same place in memory - they are listed
  separately in the subroutine call because f77 will not allow equivalencing
  pointers.  The calling routine must pass the same pointer for each.
*/

void ic4(int pk_type, float *f, struct run_param *this_run)
{
  printf("pk type : %d\n", pk_type);
  uint64_t nx = this_run->nx_tot;
  uint64_t ny = this_run->ny_tot;
  uint64_t nz = this_run->nz_tot;

  uint64_t mxs = this_run->m1s;
  uint64_t mys = this_run->m2s;
  uint64_t mzs = this_run->m3s;

  uint64_t nx2 = nx / 2;
  uint64_t ny2 = ny / 2;
  uint64_t nz2 = nz / 2;

  const uint64_t npow = 30720;

  fftwf_plan forward_planf, backward_planf;

  fftwf_init_threads();
  fftwf_plan_with_nthreads(omp_get_max_threads());

  if(this_run->itide != 0 && (mxs > 0.5 * nx || mys > 0.5 * ny || mzs > 0.5 * nz)) {

    fprintf(stderr, "Error in ic4! Subvolume must be no larger than half the size of the top grid\n");
    fprintf(stderr, "Top grid size = %lu %lu %lu\n", nx, ny, nz);
    fprintf(stderr, "Subvolume size = %lu %lu %lu\n", mxs, mys, mzs);
    exit(EXIT_FAILURE);
  }

  double dkx = TWO_PI / (nx * this_run->dx);
  double dky = TWO_PI / (ny * this_run->dx);
  double dkz = TWO_PI / (nz * this_run->dx);
  double d3k = dkx * dky * dkz;
  double akmax = TWO_PI / this_run->dx;

  /*
  Precompute transfer function table for interpolation.
  N.B. must go to at least sqrt(3)*akmax/2 unless use Hanning filter,
  in which case go to akmax/2.
  */
  double akmaxf = akmax;
  double astart = this_run->astart;

  double *tsav;
  tsav = (double *)malloc(sizeof(double) * npow);

  /* cdm */
  if(pk_type == 0) {
    for(int j = 0; j < npow; j++) {
      double ak = j * akmaxf / (double)npow;
#ifdef __OUTPUT_CDM_FOR_CB__
      tsav[j] = sqrt(calc_pk_cb(ak, astart, &this_run->cosm) * d3k);
#else
      tsav[j] = sqrt(calc_pk_cdm(ak, astart, &this_run->cosm) * d3k);
#endif
    }
  }
  /* baryon */
  else if(pk_type == 1) {
    for(int j = 0; j < npow; j++) {
      double ak = j * akmaxf / (double)npow;
      tsav[j] = sqrt(calc_pk_bar(ak, astart, &this_run->cosm) * d3k);
    }
  }
  /* neutrino 1 */
  else if(pk_type == 2) {
    for(int j = 0; j < npow; j++) {
      double ak = j * akmaxf / (double)npow;
      tsav[j] = sqrt(calc_pk_nu(ak, astart, &this_run->cosm, 1) * d3k);
    }
  }
  /* neutrino 2 */
  else if(pk_type == 3) {
    for(int j = 0; j < npow; j++) {
      double ak = j * akmaxf / (double)npow;
      tsav[j] = sqrt(calc_pk_nu(ak, astart, &this_run->cosm, 2) * d3k);
    }
  }
  /* neutrino 3 */
  else if(pk_type == 4) {
    for(int j = 0; j < npow; j++) {
      double ak = j * akmaxf / (double)npow;
      tsav[j] = sqrt(calc_pk_nu(ak, astart, &this_run->cosm, 3) * d3k);
    }
  }

#ifdef __USE_GREEN_FUNC__
  double *gk = (double *)malloc(sizeof(double) * (nx2 + 1) * (ny2 + 1) * (nz2 + 1));
  setup_GK(gk, this_run);

  /* Multiply  minus one to match the original code. */
  /* The sign of the velocities is inverted but not required. */
  if(this_run->idim == 1 || this_run->idim == 2 || this_run->idim == 3) {
    for(uint64_t ik = 0; ik < (nx2 + 1) * (ny2 + 1) * (nz2 + 1); ik++) {
      gk[ik] *= -1.0;
    }
  }

#else
  if(this_run->idim > 0) {
    for(int j = 1; j < npow; j++) {
      double ak = j * akmaxf / (double)npow;
      tsav[j] = tsav[j] / (ak * ak); // -(-1/k^2) ?
    }
  }
#endif

  /* Get white noise sample. */
  if(this_run->irand < 2) {
    fprintf(stderr, "Warning: Generating new random numbers in ic4!\n");

    /* Create and output white noise field. */
    if(this_run->irand == 1) {

      /* for 64-bit */
      unsigned long long key = this_run->iseed;
      unsigned long long length = 4;

      // unsigned long long init[4] = {0x123ULL, 0x234ULL, 0x345ULL, 0x456ULL + key};
      // init_by_array64(init, length);

      /*** When thread parallelization, the random order is indefinite. ***/
      for(uint64_t ix = 0; ix < nx; ix++) {

        unsigned long long init[4] = {0x123ULL, 0x234ULL, 0x345ULL, 0x456ULL + key + ix};
        init_by_array64(init, length);

        for(uint64_t iy = 0; iy < ny; iy++) {
          for(uint64_t iz = 0; iz < nz; iz++) {
            double xr = normalrand64();
            FF(ix, iy, iz) = xr;
          }
        }
      }

      fprintf(stderr, "Writing random numbers used in ic4 to %s.\n", this_run->noisefilename);
      FILE *noise_fp;
      noise_fp = fopen(this_run->noisefilename, "w");

      if(noise_fp == NULL) {
        fprintf(stderr, "Cannot create %s file.\n", this_run->noisefilename);
        exit(EXIT_FAILURE);
      }

      int int_tmp;
      int_tmp = nx;
      fwrite(&int_tmp, sizeof(int), 1, noise_fp);
      int_tmp = ny;
      fwrite(&int_tmp, sizeof(int), 1, noise_fp);
      int_tmp = nz;
      fwrite(&int_tmp, sizeof(int), 1, noise_fp);
      int_tmp = this_run->iseed;
      fwrite(&int_tmp, sizeof(int), 1, noise_fp);

      for(uint64_t ix = 0; ix < nx; ix++) {
        for(uint64_t iy = 0; iy < ny; iy++) {
          for(uint64_t iz = 0; iz < nz; iz++) {
            fwrite(&FF(ix, iy, iz), sizeof(float), 1, noise_fp);
          }
        }
      }

      fclose(noise_fp);
    }

  } else if(this_run->irand == 2) {
    fprintf(stderr, "Reading random numbers used in ic4 from %s.\n", this_run->noisefilename);

    FILE *noise_fp;
    noise_fp = fopen(this_run->noisefilename, "r");

    if(noise_fp == NULL) {
      fprintf(stderr, "Cannot create %s file.\n", this_run->noisefilename);
      exit(EXIT_FAILURE);
    }

    int int_tmp;
    fread(&int_tmp, sizeof(int), 1, noise_fp);
    nx = int_tmp;
    fread(&int_tmp, sizeof(int), 1, noise_fp);
    ny = int_tmp;
    fread(&int_tmp, sizeof(int), 1, noise_fp);
    nz = int_tmp;
    fread(&int_tmp, sizeof(int), 1, noise_fp);
    this_run->iseed = int_tmp;

    assert(this_run->nx_tot == nx);
    assert(this_run->ny_tot == ny);
    assert(this_run->nz_tot == nz);

    this_run->np_tot = nx * ny * nz;

    for(uint64_t ix = 0; ix < nx; ix++) {
      for(uint64_t iy = 0; iy < ny; iy++) {
        for(uint64_t iz = 0; iz < nz; iz++) {
          fread(&FF(ix, iy, iz), sizeof(float), 1, noise_fp);

          // read fortran output rand file
          // fread(&FF(iz, iy, ix), sizeof(float), 1, noise_fp);
        }
      }
    }

    fclose(noise_fp);
  }

  /* Compute mean. */
  double avg = 0.0;

#pragma omp parallel for collapse(2) reduction(+ : avg)
  for(uint64_t ix = 0; ix < nx; ix++) {
    for(uint64_t iy = 0; iy < ny; iy++) {
      for(uint64_t iz = 0; iz < nz; iz++) {
        avg += FF(ix, iy, iz);
      }
    }
  }

  avg /= (double)(nx * ny * nz);
  double fact = sqrt((double)(nx * ny * nz));
  // double fact = (double)(nx*ny*nz);
  double chisq = 0.0;
  double sigma = 0.0;

#pragma omp parallel for collapse(2) reduction(+ : chisq)
  for(uint64_t ix = 0; ix < nx; ix++) {
    for(uint64_t iy = 0; iy < ny; iy++) {
      for(uint64_t iz = 0; iz < nz; iz++) {

        /* Subtract mean. */
        FF(ix, iy, iz) -= avg;

        /* Compute chisq for this sample. */
        chisq += SQR(FF(ix, iy, iz));

        /* normalize f for FFT */
        FF(ix, iy, iz) /= fact;
      }
    }
  }

  uint64_t ndof = nx * ny * nz - 1;
  double anu = (chisq - ndof) / sqrt((double)ndof);

  printf("ic4 white noise: chisq, dof, anu, avg = %g %lu %g %g\n", chisq, ndof, anu, avg);

  fftwf_complex *f_hat;
  f_hat = (fftwf_complex *)f;
  /* f_hat(nx,ny,nz/2+1) = f_hat(nx,ny,nz/2) and f_hat(nx,ny,kz=z/2+1)
     = f(nz/2,ny,nx) and fny(ny,nx) (fny(k1=n1/2+1,ny,nx)) */

  /* Transform noise to k-space. */
  forward_planf = fftwf_plan_dft_r2c_3d(nx, ny, nz, (float *)f, (fftwf_complex *)f_hat, FFTW_ESTIMATE);
  fftwf_execute(forward_planf);
  fftwf_destroy_plan(forward_planf);

  printf("Done forward fft.\n");
  fflush(stdout);

  chisq = 0.0;
  sigma = 0.0;

  int idim = this_run->idim;
  double xoff = this_run->xoff;

  /* Generate unconstrained sample in Fourier transform space. */
#pragma omp parallel for collapse(2) reduction(+ : chisq, sigma)
  for(uint64_t kx = 0; kx < nx; kx++) {
    for(uint64_t ky = 0; ky < ny; ky++) {
      for(uint64_t kz = 0; kz < nz2 + 1; kz++) {

        double akx = (double)kx * dkx;
        if(kx > nx2) akx -= akmax;
        double akxx = akx * akx;

        double aky = (double)ky * dky;
        if(ky > ny2) aky -= akmax;
        double akyy = aky * aky;

        double akz = kz * dkz;
        if(kz == nz2) akz = 0.5 * akmax;
        double akzz = akz * akz;

        double akk = akxx + akyy + akzz;
        double ak = sqrt(akk);

        /* Evaluate transfer function. */
        double dp = (double)(npow)*ak / akmaxf;
        double tf = 0.0;

        if(dp < (double)npow) {
          int jp = (int)dp;
          dp = dp - jp;
          tf = (1.0 - dp) * tsav[jp] + dp * tsav[jp + 1];
        }

        /* Shift using offsets, with care at the Brillouin zone boundaries. */
        double theta = akz * xoff;
        if(kx != nx2) theta = theta + akx * xoff;
        if(ky != ny2) theta = theta + aky * xoff;

        struct complexd z;
        z.real = cos(theta);
        z.imag = sin(theta);

        /* These factors correctly average shifts at the Nyquist planes. */
        if(kx == nx2) {
          double tmp = cos(akx * xoff);
          z.real *= tmp;
          z.imag *= tmp;
        }
        if(ky == ny2) {
          double tmp = cos(aky * xoff);
          z.real *= tmp;
          z.imag *= tmp;
        }

        /* Convolve white noise with transfer function. */
        struct complexd tmp_fc, r0i1;
        tmp_fc.real = cmplx_re(FC(kx, ky, kz));
        tmp_fc.imag = cmplx_im(FC(kx, ky, kz));

        tmp_fc = cmplx_mul(tmp_fc, z);

        tmp_fc.real *= tf;
        tmp_fc.imag *= tf;

        r0i1.real = 0.0;
        r0i1.imag = 1.0;

        if(idim == 1) {
          tmp_fc = cmplx_mul(tmp_fc, r0i1);
          tmp_fc.real *= akx;
          tmp_fc.imag *= akx;

          if(kx == nx2) {
            tmp_fc.real = 0.0;
            tmp_fc.imag = 0.0;
          }

        } else if(idim == 2) {
          tmp_fc = cmplx_mul(tmp_fc, r0i1);
          tmp_fc.real *= aky;
          tmp_fc.imag *= aky;

          if(ky == ny2) {
            tmp_fc.real = 0.0;
            tmp_fc.imag = 0.0;
          }

        } else if(idim == 3) {
          tmp_fc = cmplx_mul(tmp_fc, r0i1);
          tmp_fc.real *= akz;
          tmp_fc.imag *= akz;

          if(kz == nz2) {
            tmp_fc.real = 0.0;
            tmp_fc.imag = 0.0;
          }
        }

#ifdef __USE_GREEN_FUNC__

        double gk_factor = 1.0;

        if(idim > 0) {
          int jkx, jky, jkz;
          if(kx < nx2) jkx = kx;
          else jkx = nx - kx;

          if(ky < ny2) jky = ky;
          else jky = ny - ky;

          jkz = kz;

          gk_factor = gk[(jkz) + (nz2 + 1) * ((jky) + (ny2 + 1) * (jkx))];
          tmp_fc.real *= gk_factor;
          tmp_fc.imag *= gk_factor;
        }
#endif // __USE_GREEN_FUNC__

        cmplx_re(FC(kx, ky, kz)) = tmp_fc.real;
        cmplx_im(FC(kx, ky, kz)) = tmp_fc.imag;

        /* Double the contribution to account for modes with k1 > n12+1 (k1 < 0). */
        int modes = 2;

        if(kz == 0 || kz == nz2) modes = 1;

        sigma += (double)modes * cmplx_abs2(tmp_fc);

#ifdef __USE_GREEN_FUNC__

        if(idim == 0) chisq += (double)modes * tf * tf;
        else if(idim == 4) chisq += (double)modes * tf * tf * gk_factor * gk_factor;
        else chisq += (double)modes * tf * tf * akk * gk_factor * gk_factor / 3.0;

#else
        if(idim == 0 || idim == 4) chisq += (double)modes * tf * tf;
        else chisq += (double)modes * tf * tf * akk / 3.0;
#endif
      }
    }
  }

  free(tsav);

#ifdef __USE_GREEN_FUNC__
  free(gk);
#endif

  if(this_run->mpi_rank == 0) {
    printf("Generate unconstrained sample.\n");
    fflush(stdout);
  }

  /* Enforce zero mean. */
  cmplx_re(FC(0, 0, 0)) = 0.0;
  cmplx_im(FC(0, 0, 0)) = 0.0;

  chisq = sqrt(chisq);
  sigma = sqrt(sigma);

  /* Transform to position space. */
  backward_planf = fftwf_plan_dft_c2r_3d(nx, ny, nz, (fftwf_complex *)f_hat, (float *)f, FFTW_ESTIMATE);

  fftwf_execute(backward_planf);
  fftwf_destroy_plan(backward_planf);

  printf("Done barckward fft.\n");
  fflush(stdout);

  double fm = 0.0;

#pragma omp parallel for reduction(max : fm)
  for(uint64_t ix = 0; ix < nx; ix++) {
    for(uint64_t iy = 0; iy < ny; iy++) {
      for(uint64_t iz = 0; iz < nz; iz++) {
        /* normalize f for FFT */
        // FF(ix,iy,iz) /= fact;
        fm = fmax(fm, fabs(FF(ix, iy, iz)));
        // printf("f %d %d %d %g %g\n",ix,iy,iz,FF(ix,iy,iz), FF(ix,iy,iz)/FF(0,0,0));
      }
    }
  }

  printf("Statistics of ic4 for idim, itide = %d %d\n", idim, this_run->itide);
  printf("Mean sigma, sampled sigma, maximum = %g %g %g\n\n", chisq, sigma, fm);
  fflush(stdout);

  fftwf_cleanup_threads();
}
