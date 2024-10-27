#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "graficlass.h"

int hunt(double *, int, double);

//#define NKBIN (201)

// same klist = np.logspace(-5, 2, 2001)
#define NKBIN (2001)

/*
input format
k tot cb cdm baryon nu[0] nu[1] nu[2]
*/

double calc_ttot_class(double ak, double a, struct cosmology *cosm)
{
  static int nkbin = NKBIN;
  static int isfirst = 1;
  static double k[NKBIN], ttot[NKBIN];
  static char buffer[512];

  if(isfirst) {
    FILE *fp;
    fp = fopen(cosm->tkfilename, "r");

    if(fp == NULL) {
      fprintf(stderr, "File %s not found in input grafic file.\n", cosm->tkfilename);
      exit(EXIT_FAILURE);
    }

    // read header
    fgets(buffer, 512, fp);

    for(int i = 0; i < nkbin; i++) {
      if(cosm->mass_nu_num == 0) {
        fscanf(fp, "%lf %lf %*lf %*lf %*lf", &k[i], &ttot[i]);
      } else if(cosm->mass_nu_num == 1) {
        fscanf(fp, "%lf %lf %*lf %*lf %*lf %*lf", &k[i], &ttot[i]);
      } else if(cosm->mass_nu_num == 2) {
        fscanf(fp, "%lf %lf %*lf %*lf %*lf %*lf %*lf", &k[i], &ttot[i]);
      } else if(cosm->mass_nu_num == 3) {
        fscanf(fp, "%lf %lf %*lf %*lf %*lf %*lf %*lf %*lf", &k[i], &ttot[i]);
      }
    }

    fclose(fp);

    isfirst = 0;
  }

  double ttot_class;

  if(ak < k[0]) {
    ttot_class = ttot[0];

  } else if(ak > k[nkbin - 1]) {
    ttot_class = ttot[nkbin - 1] + (log(ttot[nkbin - 1]) - log(ttot[nkbin - 2])) /
                                       (log(k[nkbin - 1]) - log(k[nkbin - 2])) * (log(ak) - log(k[nkbin - 1]));
  } else {
    int ilo = hunt(k, nkbin, ak);
    ttot_class = log(ttot[ilo]) +
                 (log(ttot[ilo + 1]) - log(ttot[ilo])) / (log(k[ilo + 1]) - log(k[ilo])) * (log(ak) - log(k[ilo]));

    ttot_class = exp(ttot_class);
  }

  return ttot_class;
}

double calc_tcb_class(double ak, double a, struct cosmology *cosm)
{
  const int nkbin = NKBIN;
  static int isfirst = 1;
  static double k[NKBIN], tcb[NKBIN];
  static char buffer[512];

  if(isfirst) {
    FILE *fp;
    fp = fopen(cosm->tkfilename, "r");

    if(fp == NULL) {
      fprintf(stderr, "File %s not found in input grafic file.\n", cosm->tkfilename);
      exit(EXIT_FAILURE);
    }

    // read header
    fgets(buffer, 512, fp);

    for(int i = 0; i < nkbin; i++) {
      if(cosm->mass_nu_num == 0) {
        fscanf(fp, "%lf %*lf %lf %*lf %*lf", &k[i], &tcb[i]);
      } else if(cosm->mass_nu_num == 1) {
        fscanf(fp, "%lf %*lf %lf %*lf %*lf %*lf", &k[i], &tcb[i]);
      } else if(cosm->mass_nu_num == 2) {
        fscanf(fp, "%lf %*lf %lf %*lf %*lf %*lf %*lf", &k[i], &tcb[i]);
      } else if(cosm->mass_nu_num == 3) {
        fscanf(fp, "%lf %*lf %lf %*lf %*lf %*lf %*lf %*lf", &k[i], &tcb[i]);
      }
    }

    fclose(fp);

    isfirst = 0;
  }

  double tcb_class;

  if(ak < k[0]) {
    tcb_class = tcb[0];

  } else if(ak > k[nkbin - 1]) {
    tcb_class = tcb[nkbin - 1] + (log(tcb[nkbin - 1]) - log(tcb[nkbin - 2])) / (log(k[nkbin - 1]) - log(k[nkbin - 2])) *
                                     (log(ak) - log(k[nkbin - 1]));

  } else {
    int ilo = hunt(k, nkbin, ak);
    tcb_class =
        log(tcb[ilo]) + (log(tcb[ilo + 1]) - log(tcb[ilo])) / (log(k[ilo + 1]) - log(k[ilo])) * (log(ak) - log(k[ilo]));

    tcb_class = exp(tcb_class);
  }

  return tcb_class;
}

double calc_tcdm_class(double ak, double a, struct cosmology *cosm)
{
  const int nkbin = NKBIN;
  static int isfirst = 1;
  static double k[NKBIN], tcdm[NKBIN];
  static char buffer[512];

  if(isfirst) {
    FILE *fp;
    fp = fopen(cosm->tkfilename, "r");

    if(fp == NULL) {
      fprintf(stderr, "File %s not found in input grafic file.\n", cosm->tkfilename);
      exit(EXIT_FAILURE);
    }

    // read header
    fgets(buffer, 512, fp);

    for(int i = 0; i < nkbin; i++) {
      if(cosm->mass_nu_num == 0) {
        fscanf(fp, "%lf %*lf %*lf %lf %*lf", &k[i], &tcdm[i]);
      } else if(cosm->mass_nu_num == 1) {
        fscanf(fp, "%lf %*lf %*lf %lf %*lf %*lf", &k[i], &tcdm[i]);
      } else if(cosm->mass_nu_num == 2) {
        fscanf(fp, "%lf %*lf %*lf %lf %*lf %*lf %*lf", &k[i], &tcdm[i]);
      } else if(cosm->mass_nu_num == 3) {
        fscanf(fp, "%lf %*lf %*lf %lf %*lf %*lf %*lf %*lf", &k[i], &tcdm[i]);
      }
    }

    fclose(fp);

    isfirst = 0;
  }

  double tcdm_class;

  if(ak < k[0]) {
    tcdm_class = tcdm[0];

  } else if(ak > k[nkbin - 1]) {
    tcdm_class = tcdm[nkbin - 1] + (log(tcdm[nkbin - 1]) - log(tcdm[nkbin - 2])) /
                                       (log(k[nkbin - 1]) - log(k[nkbin - 2])) * (log(ak) - log(k[nkbin - 1]));

  } else {
    int ilo = hunt(k, nkbin, ak);
    tcdm_class = log(tcdm[ilo]) +
                 (log(tcdm[ilo + 1]) - log(tcdm[ilo])) / (log(k[ilo + 1]) - log(k[ilo])) * (log(ak) - log(k[ilo]));

    tcdm_class = exp(tcdm_class);
  }

  return tcdm_class;
}

double calc_tbar_class(double ak, double a, struct cosmology *cosm)
{
  const int nkbin = NKBIN;
  static int isfirst = 1;
  static double k[NKBIN], tbar[NKBIN];
  static char buffer[512];

  if(isfirst) {
    FILE *fp;
    fp = fopen(cosm->tkfilename, "r");

    if(fp == NULL) {
      fprintf(stderr, "File %s not found in input grafic file.\n", cosm->tkfilename);
      exit(EXIT_FAILURE);
    }

    // read header
    fgets(buffer, 512, fp);

    for(int i = 0; i < nkbin; i++) {
      if(cosm->mass_nu_num == 0) {
        fscanf(fp, "%lf %*lf %*lf %*lf %lf", &k[i], &tbar[i]);
      } else if(cosm->mass_nu_num == 1) {
        fscanf(fp, "%lf %*lf %*lf %*lf %lf %*lf", &k[i], &tbar[i]);
      } else if(cosm->mass_nu_num == 2) {
        fscanf(fp, "%lf %*lf %*lf %*lf %lf %*lf %*lf", &k[i], &tbar[i]);
      } else if(cosm->mass_nu_num == 3) {
        fscanf(fp, "%lf %*lf %*lf %*lf %lf %*lf %*lf %*lf", &k[i], &tbar[i]);
      }
    }

    fclose(fp);

    isfirst = 0;
  }

  double tbar_class;

  if(ak < k[0]) {
    tbar_class = tbar[0];

  } else if(ak > k[nkbin - 1]) {
    tbar_class = tbar[nkbin - 1] + (log(tbar[nkbin - 1]) - log(tbar[nkbin - 2])) /
                                       (log(k[nkbin - 1]) - log(k[nkbin - 2])) * (log(ak) - log(k[nkbin - 1]));

  } else {
    int ilo = hunt(k, nkbin, ak);
    tbar_class = log(tbar[ilo]) +
                 (log(tbar[ilo + 1]) - log(tbar[ilo])) / (log(k[ilo + 1]) - log(k[ilo])) * (log(ak) - log(k[ilo]));

    tbar_class = exp(tbar_class);
  }

  return tbar_class;
}

double calc_tnu1_class(double ak, double a, struct cosmology *cosm)
{
  const int nkbin = NKBIN;
  static int isfirst = 1;
  static double k[NKBIN], tnu[NKBIN];
  static char buffer[512];

  if(isfirst) {
    FILE *fp;
    fp = fopen(cosm->tkfilename, "r");

    if(fp == NULL) {
      fprintf(stderr, "File %s not found in input grafic file.\n", cosm->tkfilename);
      exit(EXIT_FAILURE);
    }

    // read header
    fgets(buffer, 512, fp);

    for(int i = 0; i < nkbin; i++) {
      if(cosm->mass_nu_num == 0) {
        fprintf(stderr, "Neutrino transfer functions not found in input %s file.\n", cosm->tkfilename);
        exit(EXIT_FAILURE);
      } else if(cosm->mass_nu_num == 1) {
        fscanf(fp, "%lf %*lf %*lf %*lf %*lf %lf", &k[i], &tnu[i]);
      } else if(cosm->mass_nu_num == 2) {
        fscanf(fp, "%lf %*lf %*lf %*lf %*lf %lf %*lf", &k[i], &tnu[i]);
      } else if(cosm->mass_nu_num == 3) {
        fscanf(fp, "%lf %*lf %*lf %*lf %*lf %lf %*lf %*lf", &k[i], &tnu[i]);
      }
    }

    fclose(fp);

    isfirst = 0;
  }

  double tnu_class;

  if(ak < k[0]) {
    tnu_class = tnu[0];

  } else if(ak > k[nkbin - 1]) {
    tnu_class = tnu[nkbin - 1] + (log(tnu[nkbin - 1]) - log(tnu[nkbin - 2])) / (log(k[nkbin - 1]) - log(k[nkbin - 2])) *
                                     (log(ak) - log(k[nkbin - 1]));

  } else {
    int ilo = hunt(k, nkbin, ak);
    tnu_class =
        log(tnu[ilo]) + (log(tnu[ilo + 1]) - log(tnu[ilo])) / (log(k[ilo + 1]) - log(k[ilo])) * (log(ak) - log(k[ilo]));

    tnu_class = exp(tnu_class);
  }

  return tnu_class;
}

double calc_tnu2_class(double ak, double a, struct cosmology *cosm)
{
  const int nkbin = NKBIN;
  static int isfirst = 1;
  static double k[NKBIN], tnu[NKBIN];
  static char buffer[512];

  if(isfirst) {
    FILE *fp;
    fp = fopen(cosm->tkfilename, "r");

    if(fp == NULL) {
      fprintf(stderr, "File %s not found in input grafic file.\n", cosm->tkfilename);
      exit(EXIT_FAILURE);
    }

    // read header
    fgets(buffer, 512, fp);

    for(int i = 0; i < nkbin; i++) {
      if(cosm->mass_nu_num == 0 || cosm->mass_nu_num == 1) {
        fprintf(stderr, "Neutrino transfer functions not found in input %s file.\n", cosm->tkfilename);
        exit(EXIT_FAILURE);
      } else if(cosm->mass_nu_num == 2) {
        fscanf(fp, "%lf %*lf %*lf %*lf %*lf %*lf %lf", &k[i], &tnu[i]);
      } else if(cosm->mass_nu_num == 3) {
        fscanf(fp, "%lf %*lf %*lf %*lf %*lf %*lf %lf %*lf", &k[i], &tnu[i]);
      }
    }

    fclose(fp);

    isfirst = 0;
  }

  double tnu_class;

  if(ak < k[0]) {
    tnu_class = tnu[0];

  } else if(ak > k[nkbin - 1]) {
    tnu_class = tnu[nkbin - 1] + (log(tnu[nkbin - 1]) - log(tnu[nkbin - 2])) / (log(k[nkbin - 1]) - log(k[nkbin - 2])) *
                                     (log(ak) - log(k[nkbin - 1]));

  } else {
    int ilo = hunt(k, nkbin, ak);
    tnu_class =
        log(tnu[ilo]) + (log(tnu[ilo + 1]) - log(tnu[ilo])) / (log(k[ilo + 1]) - log(k[ilo])) * (log(ak) - log(k[ilo]));

    tnu_class = exp(tnu_class);
  }

  return tnu_class;
}

double calc_tnu3_class(double ak, double a, struct cosmology *cosm)
{
  const int nkbin = NKBIN;
  static int isfirst = 1;
  static double k[NKBIN], tnu[NKBIN];
  static char buffer[512];

  if(isfirst) {
    FILE *fp;
    fp = fopen(cosm->tkfilename, "r");

    if(fp == NULL) {
      fprintf(stderr, "File %s not found in input grafic file.\n", cosm->tkfilename);
      exit(EXIT_FAILURE);
    }

    // read header
    fgets(buffer, 512, fp);

    for(int i = 0; i < nkbin; i++) {
      if(cosm->mass_nu_num == 0 || cosm->mass_nu_num == 1 || cosm->mass_nu_num == 2) {
        fprintf(stderr, "Neutrino transfer functions not found in input %s file.\n", cosm->tkfilename);
        exit(EXIT_FAILURE);
      } else if(cosm->mass_nu_num == 3) {
        fscanf(fp, "%lf %*lf %*lf %*lf %*lf %*lf %*lf %lf", &k[i], &tnu[i]);
      }
    }

    fclose(fp);

    isfirst = 0;
  }

  double tnu_class;

  if(ak < k[0]) {
    tnu_class = tnu[0];

  } else if(ak > k[nkbin - 1]) {
    tnu_class = tnu[nkbin - 1] + (log(tnu[nkbin - 1]) - log(tnu[nkbin - 2])) / (log(k[nkbin - 1]) - log(k[nkbin - 2])) *
                                     (log(ak) - log(k[nkbin - 1]));

  } else {
    int ilo = hunt(k, nkbin, ak);
    tnu_class =
        log(tnu[ilo]) + (log(tnu[ilo + 1]) - log(tnu[ilo])) / (log(k[ilo + 1]) - log(k[ilo])) * (log(ak) - log(k[ilo]));

    tnu_class = exp(tnu_class);
  }

  return tnu_class;
}

int hunt(double *xx, int n, double x)
{
  int jlo = -1;

  if(xx[n - 1] >= xx[0]) {

    for(int i = 0; i < n - 1; i++) {
      if(x >= xx[i] && x <= xx[i + 1]) {
        jlo = i;
        break;
      }
    }

  } else {

    for(int i = 1; i < n; i++) {
      if(x <= xx[i - 1] && x >= xx[i]) {
        jlo = i - 1;
        break;
      }
    }
  }

  return jlo;
}
