#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "graficlass.h"
#include "constants.h"
#include "prototype.h"

void read_input_graficlassfile(char *inputfile, struct run_param *this_run)
{
  FILE *fp;
  fp = fopen(inputfile, "r");

  if(fp == NULL) {
    fprintf(stderr, "File %s not found in input grafic file.\n", inputfile);
    exit(EXIT_FAILURE);
  }

  printf("input file : %s\n", inputfile);

  /* set the tk and pk file names */
  this_run->tkfilename = (char *)malloc(sizeof(char) * 512);
  this_run->pkfilename = (char *)malloc(sizeof(char) * 512);

  fscanf(fp, "%s", this_run->tkfilename);
  fscanf(fp, "%s", this_run->pkfilename);

  if(this_run->mpi_rank == 0) {
    printf("tk filename : %s\n", this_run->tkfilename);
    printf("pk filename : %s\n", this_run->pkfilename);
  }

  this_run->cosm.tkfilename = this_run->tkfilename;
  this_run->cosm.pkfilename = this_run->pkfilename;

  /*
    Select type of initial power spectrum (matter transfer function)
    1 for T(k) from linger.dat
    2 for T(k) from approx. analytical fit
    3 for T(k)=1 (scale-free)
  */
  fscanf(fp, "%d", &(this_run->icase));

  if(this_run->mpi_rank == 0) printf("icase : %d\n", this_run->icase);

  this_run->cosm.icase = this_run->icase;

  if(this_run->icase != 2) {
    fprintf(stderr, "~~~ This c version supports only '2'. ~~~\n");
    exit(EXIT_FAILURE);
  }

  /* set omegam,omegav,omegab,h0,omega_rad */
  fscanf(fp, "%lf,  %lf, %lf, %lf, %lf", &(this_run->cosm.omegam), &(this_run->cosm.omegav), &(this_run->cosm.omegab),
         &(this_run->cosm.h0), &(this_run->cosm.omegar));

  if(this_run->mpi_rank == 0)
    printf("Omegam, Omegav, Omegab, Omegarad, H0 : %g %g %g %g %g\n", this_run->cosm.omegam, this_run->cosm.omegav,
           this_run->cosm.omegab, this_run->cosm.omegar, this_run->cosm.h0);

  /* long-wave spectral index n (scale-invariant is n=1) */
  fscanf(fp, "%lf", &(this_run->cosm.an));

  if(this_run->mpi_rank == 0) printf("n_s : %g\n", this_run->cosm.an);

  /*
    desired normalization at a=1
    Qrms-ps/micro-K (if > 0) or -sigma_8 (if < 0) = ?
  */
  fscanf(fp, "%lf", &(this_run->cosm.anorml));

  if(this_run->mpi_rank == 0) printf("anorml : %g\n", this_run->cosm.anorml);

  /*
    kmin and kmax (1/Mpc)'
  */
  fscanf(fp, "%lf, %lf", &(this_run->ak1), &(this_run->ak2));

  if(this_run->mpi_rank == 0) printf("kmin, kmax : %g %g [1/Mpc]\n", this_run->ak1, this_run->ak2);

  this_run->cosm.akmin = this_run->ak1;
  this_run->cosm.akmax = this_run->ak2;

  /* dx (initial particle spacing in Mpc, not Mpc/h)'
     or enter -boxlength in Mpc/h'
     i.e. dx in Mpc if > 0, or -boxlength in Mpc/h if < 0'
  */
  fscanf(fp, "%lf", &(this_run->dx));

  if(this_run->mpi_rank == 0) {
    if(this_run->dx < 0.0) printf("box size : %g [Mpc/h]\n", this_run->dx);
    else printf("grid size : %g [Mpc]\n", this_run->dx);
  }

  /* initial redshift */
  fscanf(fp, "%lf", &(this_run->init_zred));

  if(this_run->mpi_rank == 0) printf("init_zred : %g\n", this_run->init_zred);

  /* neutrino particle mass in units of eV */
  fscanf(fp, "%d", &(this_run->mass_nu_num));

  for(int inu = 0; inu < MAX_NU_NUM; inu++) {
    this_run->mass_nu_deg[inu] = 0;
    this_run->mass_nu_mass[inu] = 0.0;
    this_run->mass_nu_frac[inu] = 0.0;
  }

  if(this_run->mass_nu_num == 0) {
    fscanf(fp, "%d", &(this_run->mass_nu_deg[0]));
    fscanf(fp, "%lf", &(this_run->mass_nu_mass[0]));
    fscanf(fp, "%lf", &(this_run->mass_nu_frac[0]));
  } else if(this_run->mass_nu_num == 1) {
    fscanf(fp, "%d", &(this_run->mass_nu_deg[0]));
    fscanf(fp, "%lf", &(this_run->mass_nu_mass[0]));
    fscanf(fp, "%lf", &(this_run->mass_nu_frac[0]));
  } else if(this_run->mass_nu_num == 2) {
    fscanf(fp, "%d,%d", &(this_run->mass_nu_deg[0]), &(this_run->mass_nu_deg[1]));
    fscanf(fp, "%lf,%lf", &(this_run->mass_nu_mass[0]), &(this_run->mass_nu_mass[1]));
    fscanf(fp, "%lf,%lf", &(this_run->mass_nu_frac[0]), &(this_run->mass_nu_frac[1]));
  } else if(this_run->mass_nu_num == 3) {
    fscanf(fp, "%d,%d,%d", &(this_run->mass_nu_deg[0]), &(this_run->mass_nu_deg[1]), &(this_run->mass_nu_deg[2]));
    fscanf(fp, "%lf,%lf,%lf", &(this_run->mass_nu_mass[0]), &(this_run->mass_nu_mass[1]), &(this_run->mass_nu_mass[2]));
    fscanf(fp, "%lf,%lf,%lf", &(this_run->mass_nu_frac[0]), &(this_run->mass_nu_frac[1]), &(this_run->mass_nu_frac[2]));
  }

  double nu_mass_tot = 0.0;
  for(int inu = 0; inu < MAX_NU_NUM; inu++) {
    nu_mass_tot += this_run->mass_nu_deg[inu] * this_run->mass_nu_mass[inu];
  }

  this_run->nu_mass_tot = nu_mass_tot;
  this_run->cosm.mass_nu_num = this_run->mass_nu_num;

  if(this_run->mpi_rank == 0) {
    printf("massive neutrino num : %d\n", this_run->mass_nu_num);
    printf("total neutrino mass : %g [eV]\n", nu_mass_tot);
    printf("massive neutrino degeneracy : %d, %d, %d\n", this_run->mass_nu_deg[0], this_run->mass_nu_deg[1],
           this_run->mass_nu_deg[2]);
    printf("massive neutrino mass : %g, %g, %g [eV]\n", this_run->mass_nu_mass[0], this_run->mass_nu_mass[1],
           this_run->mass_nu_mass[2]);
    printf("massive neutrino fraction : %.4f, %.4f, %.4f\n", this_run->mass_nu_frac[0], this_run->mass_nu_frac[1],
           this_run->mass_nu_frac[2]);
  }

  /*
    the ultimate refinement factor for the smallest grid spacing
    (This is used only to set astart.)
  */
  fscanf(fp, "%lf", &(this_run->nrefine));
  if(this_run->mpi_rank == 0) printf("refinement factor : %g\n", this_run->nrefine);
  if(this_run->nrefine < 1.0) {
    fprintf(stderr, "Error! refinement factor must be >= 1\n");
    exit(EXIT_FAILURE);
  }

  /*
    Now set output parameters.  There are two cases:
    hanning=T: no further refinement.
    hanning=F: prepare for further refinement.
    Enter 0 for final output or 1 for further refinement
*/
  int iref;
  fscanf(fp, "%d", &iref);

  if(this_run->mpi_rank == 0) printf("iref : %d\n", iref);
  if(iref == 1) {
    fprintf(stderr, "iref=1 is not supported in this version.\n");
    exit(EXIT_FAILURE);
  }

  if(iref == 0) {
    this_run->hanning = 1; //.true.
    this_run->m1s = this_run->nx_tot;
    this_run->m2s = this_run->ny_tot;
    this_run->m3s = this_run->nz_tot;
    this_run->x1o = 0.0;
    this_run->x2o = 0.0;
    this_run->x3o = 0.0;

    if(this_run->mpi_rank == 0)
      printf("setting output grid to %lu %lu %lu\n", this_run->m1s, this_run->m2s, this_run->m3s);

    /*
      skip four lines :
      output grid size, output grid offset,
      final grid size, final grid offset
    */

    /*
    char dummy[64];
    fscanf(fp,"%*s", dummy);
    fscanf(fp,"%*s", dummy);
    fscanf(fp,"%*s", dummy);
    fscanf(fp,"%*s", dummy);
    */
  }

  /* Set parameters for subgrid noise.
     Choose irand (1 or 2) from the following list:
     irand=0 to generate new noise and don''t save it
     irand=1 to generate new noise and save to file
     irand=2 to read noise from existing file
  */
  fscanf(fp, "%d", &(this_run->irand));

  if(this_run->irand < 0 || this_run->irand > 2) {
    fprintf(stderr, "Illegal value of irand\n");
    exit(EXIT_FAILURE);
  }

  /* random number seed (9-digit integer, ignored if irand=2) */
  fscanf(fp, "%d", &(this_run->iseed));

  this_run->noisefilename = (char *)malloc(sizeof(char) * 256);
  fscanf(fp, "%s", this_run->noisefilename);

  if(this_run->mpi_rank == 0) {
    printf("irand : %d\n", this_run->irand);
    printf("iseed : %d\n", this_run->iseed);
    printf("noise file (irand=1:output, irand=2:input) : %s\n", this_run->noisefilename);
  }
  fclose(fp);
}

void set_params(struct run_param *this_run)
{
  /* box size [Mpc/h] to grid size [Mpc] */
  if(this_run->dx < 0.0) this_run->dx = -this_run->dx * 100.0 / (this_run->cosm.h0 * this_run->nx_tot);

  this_run->dxr = this_run->dx / this_run->nrefine;

  this_run->cosm.dx = this_run->dx;
  this_run->cosm.asig = 1.0e0;

  double int_min = 0.0;
  double int_max = 0.5 * TWO_PI / this_run->dxr;
  double int_err = 1.0e-7;
  this_run->sigma = 2.0 * TWO_PI * calc_rombin(calc_dsigma, int_min, int_max, int_err, &this_run->cosm);

  /* This is sigma at a=1. */
  this_run->sigma = sqrt(this_run->sigma);

  /*
    Normalize so that rms density flutuation=sigstart at starting
    redshift scaling back the fluctuations as if they grew like cdm.
  */
  double dpls = this_run->sigstart / this_run->sigma * calc_dplus(1.0, &this_run->cosm);
  double astart = 1.0 / (1.0 + this_run->init_zred);
  this_run->astart = astart;
  this_run->cosm.asig = astart;
  this_run->sigma = 2.0 * TWO_PI * calc_rombin(calc_dsigma, int_min, int_max, int_err, &this_run->cosm);
  this_run->sigma = sqrt(this_run->sigma);

  if(this_run->mpi_rank == 0) {
    printf("Scaling initial conditions to starting a = %g , redshift z = %g\n", astart, 1.0 / astart - 1.0);
    printf("when sigma at ultimate refinement scale = %g\n", this_run->sigma);
  }

  /*
    velocity (proper km/s) =  Displacement (comoving Mpc at astart) * vfact.
    vfact = dln(D+)/dtau where tau=conformal time.
  */
  this_run->vfact =
      calc_fomega(astart, &this_run->cosm) * this_run->cosm.h0 * calc_dladt(astart, &this_run->cosm) / astart;
  this_run->vfact_2LPT =
      calc_fomega_2LPT(astart, &this_run->cosm) * this_run->cosm.h0 * calc_dladt(astart, &this_run->cosm) / astart;

  if(this_run->mpi_rank == 0) {
    printf("vfact = %g\n", this_run->vfact);
    printf("vfact_2LPT = %g\n", this_run->vfact_2LPT);
  }
}
