#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "graficlass.h"
#include "prototype.h"
#include "constants.h"

/*
  This function calculates the variance of density with a Hanning
  filter at the grid Nyquist frequency.
*/
double calc_dsigma(double ak, struct cosmology *cosm)
{
  double dsigma = 0.0;
  double dx = cosm->dx;
  double asig = cosm->asig;

  if(ak < 0.0 || ak >= PI / dx) return dsigma;

  /* Hanning filter. */
  dsigma = ak * ak * calc_p(ak, asig, cosm) * cos(0.5 * ak * dx);
  return dsigma;
}

/*
  Pini initializes the power spectrum.
  common /cosmoparms/ omegam,omegav,omegab,h0
  common /pstuff/ atab,an,pnorm,icase,ilog,ntab
  common /omegas/ om,ov,ok
  common /phint/ tcon0,ak
  common /splin1/ deltat2,ddeltat2,dk,akminl
  common /splin2/ tmat,dtmat,tcdm,dtcdm,tbar,dtbar,dk1,akmin1,nk1
  common /scaling/ a00,a10,scale
  common /extwarn/ iwarn1,iwarn2
  common /powerfile/ pkfilename
*/
void pini(struct run_param *this_run)
{
  int icase = this_run->icase;

  this_run->iwarn1 = 0;
  this_run->iwarn2 = 0;

  if(icase == 1 || icase == 3) {
    fprintf(stderr, "icase=1 or 3 is not supported in this version.\n");
    exit(EXIT_FAILURE);
  } else if(icase == 2) {
    this_run->tcmb = 2.726e6;

    this_run->cosm.omegak = 1.0e0 - this_run->cosm.omegam - this_run->cosm.omegav - this_run->cosm.omegar;
    if(fabs(this_run->cosm.omegak) < 1.0e-5) this_run->cosm.omegak = 0;

    this_run->ntab = NT;
    this_run->atab[0] = 1.0;

  } else {
    fprintf(stderr, "Illegal choice.\n");
    exit(EXIT_FAILURE);
  }

  struct cosmology *cosm = &(this_run->cosm);

  cosm->a00 = 0.0;
  cosm->a10 = 0.0;
  cosm->scale = 1.0;

  /*
    Transfer function case.  Normalize by CMB quadrupole.
    Two subcases: BBKS transfer function or linger.dat.
    First, get l=2 CMB transfer function Delta_2(k).
  */
  int nk, ilog;
  double dk;

  if(icase == 2) {
    /* Compute Delta_2(k) using Sachs-Wolfe approximation (including
     * phidot). */
    nk = 301;
    cosm->akmin = 1.e-5;
    cosm->akmax = 1.e-2;

    ilog = 1;
    dk = log(cosm->akmax / cosm->akmin) / (nk - 1);
    cosm->dk = dk;

    double tcon0 = calc_tcon(1.0, cosm);

    double arec = 1.0e0 / 1.2e3;
    double f0 = calc_dplus(1.0, cosm);
    double frec = calc_dplus(arec, cosm) / arec;

    if(this_run->mpi_rank == 0)
      printf("Computing Delta_2(k) using Sachs-Wolfe approxmation. This "
             "may take several minutes.\n");

    for(int ik = 0; ik < nk; ik++) {
      double ak = cosm->akmin * exp((ik)*dk);
      cosm->ak = ak;
      double phidotint = calc_rombin(calc_dphid, arec, 1.0, 1.0e-4, cosm);

      /*
        Assume isentropic initial fluctuations.  If they are instead
        entropy, replace uj2/3 by uj2*2.
      */
      cosm->deltat2[ik] =
          (frec * calc_uj2(ak * 2.99793e5 / cosm->h0, tcon0, cosm->omegak) / 3.0 + 2.0 * phidotint) / f0;
    }
  } // if(icase==2)

  /* Now integrate anisotropy to normalize by Qrms-ps. */
  const int nspl = 100001;
  double *gspl;
  gspl = (double *)malloc(sizeof(double) * nspl);
  splini(gspl, nspl);

  double qq;

  if(ilog == 0) {
    cosm->deltat2[0] = 0.0;
    fit_splder(cosm->deltat2, cosm->ddeltat2, gspl, nk);
    qq = 5.0 * FOUR_PI * calc_rombin(calc_dc2, 0.0, cosm->akmax, 1.0e-7, cosm);

  } else {
    fit_splder(&(cosm->deltat2[1]), &(cosm->ddeltat2[1]), gspl, nk);
    cosm->akminl = log(cosm->akmin);
    cosm->akmaxl = log(cosm->akmax);
    qq = 5.0 * FOUR_PI * calc_rombin(calc_dc2l, log(1.0e-6), cosm->akmaxl, 1.0e-7, cosm);
  }

  /*
    pnorm is the primeval amplitude defined by P_psi=pnorm*ak**(an-4)
    in the isentropic case.  For isocurvature initial conditions,
    replace P_psi by the power spectrum of primeval entropy perturbations.
  */
  cosm->akmax = cosm->h0 / 8.0;
  for(int it = 0; it < this_run->ntab; it++) {
    fit_splder(&(cosm->tmat[it][1]), &(cosm->dtmat[it][1]), gspl, nk);
    fit_splder(&(cosm->tcdm[it][1]), &(cosm->dtcdm[it][1]), gspl, nk);
    fit_splder(&(cosm->tbar[it][1]), &(cosm->dtbar[it][1]), gspl, nk);
  }

  if(cosm->anorml >= 0.0) {
    /* anorml is Qrms-ps in micro-K.  Compute corresponding sigma8. */
    cosm->pnorm = SQR(cosm->anorml / this_run->tcmb) / qq;
    /* Now integrate density fluctuation to get sigma8. */
    double sig0 = FOUR_PI * calc_rombin(calc_dsig8, 0.0, cosm->akmax, 1.0e-8, cosm);
    double sigma8 = sqrt(sig0);
    if(this_run->mpi_rank == 0) printf("Linear sigma8=%g\n", sigma8);

  } else {
    /* anorml is -sigma8, the rms linear density fluctuation in a sphere of
     */
    /* radius 8/h Mpc.  Compute corresponding Qrms-ps. */
    double sigma8 = -cosm->anorml;
    cosm->pnorm = 1.0;
    double sig0 = FOUR_PI * calc_rombin(calc_dsig8, 0.0, cosm->akmax, 1.0e-8, cosm);
    cosm->pnorm = SQR(sigma8) / sig0;
    double qrmsps = this_run->tcmb * sqrt(cosm->pnorm * qq);
    if(this_run->mpi_rank == 0) printf("Qrms-ps/micro-K=%g , %g %g %g\n", qrmsps, sig0, cosm->pnorm, qq);
  }

  free(gspl);

  int nkplot = 401;
  double dlkp = log(this_run->ak2 / this_run->ak1) / (nkplot - 1);

  if(this_run->mpi_rank == 0) {
    FILE *fp_pk;
    fp_pk = fopen(this_run->pkfilename, "w");
    fprintf(fp_pk, "# an, anorml, H0 : %g %g %g\n", this_run->cosm.an, this_run->cosm.anorml, cosm->h0);
    fprintf(fp_pk, "# k [h/Mpc], Pk_tot, Pk_cb, Pk_cdm, Pk_bar, Pk_nu1, Pk_nu2, Pk_nu3 [Mpc^3 h^-3]\n");

    /*
    These units of output are common units.
    They are different from the original grafic output power spectrum.
    */

    double anow = 1.0;

    for(int i = 0; i < nkplot; i++) {
      double ak0 = this_run->ak1 * exp((i)*dlkp);

      if(cosm->mass_nu_num == 0) {
        fprintf(fp_pk, "%16.10e %16.10e %16.10e %16.10e %16.10e\n", ak0 / (cosm->h0 / 100.0),
                calc_pk_tot(ak0, anow, cosm) * cosm->h0, calc_pk_cb(ak0, anow, cosm) * cosm->h0,
                calc_pk_cdm(ak0, anow, cosm) * cosm->h0, calc_pk_bar(ak0, anow, cosm) * cosm->h0);

      } else if(cosm->mass_nu_num == 1) {
        fprintf(fp_pk, "%16.10e %16.10e %16.10e %16.10e %16.10e %16.10e\n", ak0 / (cosm->h0 / 100.0),
                calc_pk_tot(ak0, anow, cosm) * cosm->h0, calc_pk_cb(ak0, anow, cosm) * cosm->h0,
                calc_pk_cdm(ak0, anow, cosm) * cosm->h0, calc_pk_bar(ak0, anow, cosm) * cosm->h0,
                calc_pk_nu(ak0, anow, cosm, 1) * cosm->h0);

      } else if(cosm->mass_nu_num == 2) {
        fprintf(fp_pk, "%16.10e %16.10e %16.10e %16.10e %16.10e %16.10e %16.10e\n", ak0 / (cosm->h0 / 100.0),
                calc_pk_tot(ak0, anow, cosm) * cosm->h0, calc_pk_cb(ak0, anow, cosm) * cosm->h0,
                calc_pk_cdm(ak0, anow, cosm) * cosm->h0, calc_pk_bar(ak0, anow, cosm) * cosm->h0,
                calc_pk_nu(ak0, anow, cosm, 1) * cosm->h0, calc_pk_nu(ak0, anow, cosm, 2) * cosm->h0);

      } else if(cosm->mass_nu_num == 3) {
        fprintf(fp_pk, "%16.10e %16.10e %16.10e %16.10e %16.10e %16.10e %16.10e %16.10e\n", ak0 / (cosm->h0 / 100.0),
                calc_pk_tot(ak0, anow, cosm) * cosm->h0, calc_pk_cb(ak0, anow, cosm) * cosm->h0,
                calc_pk_cdm(ak0, anow, cosm) * cosm->h0, calc_pk_bar(ak0, anow, cosm) * cosm->h0,
                calc_pk_nu(ak0, anow, cosm, 1) * cosm->h0, calc_pk_nu(ak0, anow, cosm, 2) * cosm->h0,
                calc_pk_nu(ak0, anow, cosm, 3) * cosm->h0);
      }
    }

    fclose(fp_pk);

    printf("output power file : %s\n", this_run->pkfilename);
    printf("done pini\n");
  }
}

/*
  common /cosmoparms/ omegam,omegav,omegab,h0
  common /pstuff/ atab,an,pnorm,icase,ilog,ntab
  common /scaling/ a00,a10,scale
 */
double calc_pk_tot(double ak, double a, struct cosmology *cosm)
{
  double pk_tot = 0.0;

  if(cosm->icase == 2) {

    if(ak <= 0.0) return pk_tot;

    double omega = cosm->omegam + cosm->omegav;
    double t = calc_ttot_class(ak, a, cosm);

    pk_tot = cosm->pnorm * pow(ak, (cosm->an - 4.0));
    pk_tot = pk_tot * t * t;

    /*
      Convert to density fluctuation power spectrum.  Note that k^2 is
      corrected for an open universe.  Scale to a using linear theory.
    */
    double tpois = -(2.0 / 3.0) / cosm->omegam * (SQR(ak * 2.99793e5 / cosm->h0) - 4.0 * (omega - 1.0));

    // (a10 != a) || (a00 != 1.0)
    if((cosm->a10 < a * 0.9999 || cosm->a10 > a * 1.00001) || (cosm->a00 < 0.9999 || cosm->a00 > 1.00001)) {

      cosm->a10 = a;
      cosm->a00 = 1.0;
      cosm->scale = calc_dplus(a, cosm) / calc_dplus(1.0, cosm);
    }

    pk_tot *= SQR(tpois) * SQR(cosm->scale);
  } else {
    fprintf(stderr, "Error: icase should be 2 in pk_tot");
    exit(EXIT_FAILURE);
  }
  return pk_tot;
}

double calc_pk_cb(double ak, double a, struct cosmology *cosm)
{
  double pk_cb = 0.0;

  if(cosm->icase == 2) {

    if(ak <= 0.0) return pk_cb;

    double omega = cosm->omegam + cosm->omegav;
    double t = calc_tcb_class(ak, a, cosm);

    pk_cb = cosm->pnorm * pow(ak, (cosm->an - 4.0));
    pk_cb = pk_cb * t * t;

    /*
      Convert to density fluctuation power spectrum.  Note that k^2 is
      corrected for an open universe.  Scale to a using linear theory.
    */
    double tpois = -(2.0 / 3.0) / cosm->omegam * (SQR(ak * 2.99793e5 / cosm->h0) - 4.0 * (omega - 1.0));

    // (a10 != a) || (a00 != 1.0)
    if((cosm->a10 < a * 0.9999 || cosm->a10 > a * 1.00001) || (cosm->a00 < 0.9999 || cosm->a00 > 1.00001)) {

      cosm->a10 = a;
      cosm->a00 = 1.0;
      cosm->scale = calc_dplus(a, cosm) / calc_dplus(1.0, cosm);
    }

    pk_cb *= SQR(tpois) * SQR(cosm->scale);

  } else {
    fprintf(stderr, "Error: icase should be 2 in pk_cb");
    exit(EXIT_FAILURE);
  }

  return pk_cb;
}

/*
  common /cosmoparms/ omegam,omegav,omegab,h0
  common /pstuff/ atab,an,pnorm,icase,ilog,ntab
  common /scaling/ a00,a10,scale
 */
double calc_pk_cdm(double ak, double a, struct cosmology *cosm)
{
  double pk_cdm = 0.0;

  if(cosm->icase == 2) {

    if(ak <= 0.0) return pk_cdm;

    double omega = cosm->omegam + cosm->omegav;
    double t = calc_tcdm_class(ak, a, cosm);

    pk_cdm = cosm->pnorm * pow(ak, (cosm->an - 4.0));
    pk_cdm = pk_cdm * t * t;

    /*
      Convert to density fluctuation power spectrum.  Note that k^2 is
      corrected for an open universe.  Scale to a using linear theory.
    */
    double tpois = -(2.0 / 3.0) / cosm->omegam * (SQR(ak * 2.99793e5 / cosm->h0) - 4.0 * (omega - 1.0));

    // (a10 != a) || (a00 != 1.0)
    if((cosm->a10 < a * 0.9999 || cosm->a10 > a * 1.00001) || (cosm->a00 < 0.9999 || cosm->a00 > 1.00001)) {

      cosm->a10 = a;
      cosm->a00 = 1.0;
      cosm->scale = calc_dplus(a, cosm) / calc_dplus(1.0, cosm);
    }

    pk_cdm *= SQR(tpois) * SQR(cosm->scale);

  } else {
    fprintf(stderr, "Error: icase should be 2 in pk_cdm");
    exit(EXIT_FAILURE);
  }

  return pk_cdm;
}

/*
  common /cosmoparms/ omegam,omegav,omegab,h0
  common /pstuff/ atab,an,pnorm,icase,ilog,ntab
  common /scaling/ a00,a10,scale
 */
double calc_pk_bar(double ak, double a, struct cosmology *cosm)
{
  double pk_bar = 0.0;

  if(cosm->icase == 2) {

    if(ak <= 0.0) return pk_bar;

    double omega = cosm->omegam + cosm->omegav;
    double t = calc_tbar_class(ak, a, cosm);

    pk_bar = cosm->pnorm * pow(ak, (cosm->an - 4.0));
    pk_bar = pk_bar * t * t;

    /*
      Convert to density fluctuation power spectrum.  Note that k^2 is
      corrected for an open universe.  Scale to a using linear theory.
    */
    double tpois = -(2.0 / 3.0) / cosm->omegam * (SQR(ak * 2.99793e5 / cosm->h0) - 4.0 * (omega - 1.0));

    // (a10 != a) || (a00 != 1.0)
    if((cosm->a10 < a * 0.9999 || cosm->a10 > a * 1.00001) || (cosm->a00 < 0.9999 || cosm->a00 > 1.00001)) {

      cosm->a10 = a;
      cosm->a00 = 1.0;
      cosm->scale = calc_dplus(a, cosm) / calc_dplus(1.0, cosm);
    }

    pk_bar *= SQR(tpois) * SQR(cosm->scale);

  } else {
    fprintf(stderr, "Error: icase should be 2 in pk_bar");
    exit(EXIT_FAILURE);
  }

  return pk_bar;
}

/*
  common /cosmoparms/ omegam,omegav,omegab,h0
  common /pstuff/ atab,an,pnorm,icase,ilog,ntab
  common /scaling/ a00,a10,scale
 */

double calc_pk_nu(double ak, double a, struct cosmology *cosm, int inu)
{
  double pk_nu = 0.0;

  if(cosm->icase == 2) {

    if(ak <= 0.0) return pk_nu;

    double omega = cosm->omegam + cosm->omegav;
    double t;

    if(inu == 1) {
      t = calc_tnu1_class(ak, a, cosm);
    } else if(inu == 2) {
      t = calc_tnu2_class(ak, a, cosm);
    } else if(inu == 3) {
      t = calc_tnu3_class(ak, a, cosm);
    }

    pk_nu = cosm->pnorm * pow(ak, (cosm->an - 4.0));
    pk_nu = pk_nu * t * t;

    /*
      Convert to density fluctuation power spectrum.  Note that k^2 is
      corrected for an open universe.  Scale to a using linear theory.
    */
    double tpois = -(2.0 / 3.0) / cosm->omegam * (SQR(ak * 2.99793e5 / cosm->h0) - 4.0 * (omega - 1.0));

    // (a10 != a) || (a00 != 1.0)
    if((cosm->a10 < a * 0.9999 || cosm->a10 > a * 1.00001) || (cosm->a00 < 0.9999 || cosm->a00 > 1.00001)) {

      cosm->a10 = a;
      cosm->a00 = 1.0;
      cosm->scale = calc_dplus(a, cosm) / calc_dplus(1.0, cosm);
    }

    pk_nu *= SQR(tpois) * SQR(cosm->scale);

  } else {
    fprintf(stderr, "Error: icase should be 2 in pk_nu");
    exit(EXIT_FAILURE);
  }

  return pk_nu;
}

/*
  p evaluates the power spectrum at wavenumber ak for expansion factor a.
  It takes the nearest transfer function and scales it to a using the
  cdm transfer function.
  N.B. p is the 3-D spectral density and has units of 1/(ak*ak*ak).
  N.B. ak has units of 1/Mpc, _not_ h/Mpc.

  common /cosmoparms/ omegam,omegav,omegab,h0
  common /pstuff/ atab,an,pnorm,icase,ilog,ntab
  common /splin2/ y,dy,y1,dy1,y2,dy2,dk,akminl,nk
  common /scaling/ a00,a10,scale
*/
double calc_p(double ak, double a, struct cosmology *cosm)
{
  double p = 0.0;

  if(ak <= 0.0) return p;

  double omega = cosm->omegam + cosm->omegav;

  /*
    Transfer function case.  The transfer function T(k) is defined so
    that for initial curvature perturbations, phi(k,a=1)=T(k)*psi(k,a=0)
    where psi(k,a=0) has power spectrum pnorm*ak**(an-4) and phi(k,a=1)
    is related to delta (actually, Bardeen's gauge-invariant variable
    epsilon_m) by the Poisson equation.  For isocurvature initial
    conditions, linger uses the initial entropy perturbation rather than
    psi for normalization, but everything in this subroutine and in pini
    below follows without  change.
    Two subcases: BBKS transfer function or tabulated transfer function
    from linger.dat.
  */
  if(cosm->icase == 2) {
    /*
      Use fit to matter transfer function.
      Hubble constant in units of 100 km/sec/Mpc.
    */

    double h = cosm->h0 / 100.0;
    double omegahh = cosm->omegam * h * h * exp(-cosm->omegab * (1.0 + sqrt(2.0 * h) / cosm->omegam));
    double q = ak / omegahh;

    double t = calc_tcdm_class(ak, a, cosm);

    /*
      Apply transfer function to primordial power spectrum.
      Primordial spectrum of psi (or entropy, in the isocurvature case):
   */
    double p = cosm->pnorm * pow(ak, (cosm->an - 4.0));

    /* Apply transfer function to get spectrum of phi at a=1. */
    p = p * t * t;

    /*
      Convert to density fluctuation power spectrum.  Note that k^2 is
      corrected for an open universe.  Scale to a using linear theory.
    */
    double tpois = -(2.0 / 3.0) / cosm->omegam * (SQR(ak * 2.99793e5 / cosm->h0) - 4.0 * (omega - 1.0));

    // (a10 != a) || (a00 != 1.0)
    if((cosm->a10 < a * 0.9999 || cosm->a10 > a * 1.00001) || (cosm->a00 < 0.9999 || cosm->a00 > 1.00001)) {

      cosm->a10 = a;
      cosm->a00 = 1.0;
      cosm->scale = calc_dplus(a, cosm) / calc_dplus(1.0, cosm);
    }

    p *= SQR(tpois) * SQR(cosm->scale);
    return p;
  }

  return -1.0;
}

/*
  common /splin1/ y,dy,dk,akminl

  y=deltat2
  dy=ddeltat2
 */
double calc_dc2(double ak, struct cosmology *cosm)
{
  const double tiny = 1.0e-20;

  double dc2 = 0.0;

  if(ak > -tiny && ak < tiny) return dc2;

  double d = ak / cosm->dk;
  int i = d;
  d = d - (double)i;

  double delt2 =
      cosm->deltat2[i] +
      d * (cosm->ddeltat2[i] +
           d * (3.0 * (cosm->deltat2[i + 1] - cosm->deltat2[i]) - 2.0 * cosm->ddeltat2[i] - cosm->ddeltat2[i + 1] +
                d * (cosm->ddeltat2[i] + cosm->ddeltat2[i + 1] + 2.0 * (cosm->deltat2[i] - cosm->deltat2[i + 1]))));

  dc2 = SQR(delt2) * pow(ak, (cosm->an - 2.0));

  return dc2;
}

/*
  common /splin1/ y,dy,dkl,akminl
  common /pstuff/ atab,an,pnorm,icase,ilog,ntab

  y=deltat2
  dy=ddeltat2
 */
double calc_dc2l(double akl, struct cosmology *cosm)
{
  double dkl = cosm->dk;
  double ak = exp(akl);
  double d = (akl - cosm->akminl) / dkl;

  int i = d;
  d = d - (double)i;

  double delt2, dc2l;

  if(i < 1) delt2 = cosm->deltat2[0] * SQR(ak / exp(cosm->akminl));
  else
    delt2 =
        cosm->deltat2[i] +
        d * (cosm->ddeltat2[i] +
             d * (3.0 * (cosm->deltat2[i + 1] - cosm->deltat2[i]) - 2.0 * cosm->ddeltat2[i] - cosm->ddeltat2[i + 1] +
                  d * (cosm->ddeltat2[i] + cosm->ddeltat2[i + 1] + 2.0 * (cosm->deltat2[i] - cosm->deltat2[i + 1]))));

  dc2l = SQR(delt2) * pow(ak, (cosm->an - 1.0));

  return dc2l;
}

/*
  common /cosmoparms/ omegam,omegav,omegab,h0
  common /phint/ tcon0,ak
 */
double calc_dphid(double a, struct cosmology *cosm)
{
  double ak = cosm->ak;
  double tcon0 = calc_tcon(1.0, cosm);
  double r = tcon0 - calc_tcon(a, cosm);
  double dphid = calc_dplus(a, cosm) / (a * a) * (calc_fomega(a, cosm) - 1.0) *
                 calc_uj2(ak * 2.99793e5 / cosm->h0, r, cosm->omegak);

  return dphid;
}

/*
  Omegam := Omega today (a=1) in matter.
  Omegav := Omega today (a=1) in vacuum energy.
  Omegak := Omega today (a=1) in curvature, Omegak := 1-Omegam-Omegav.
*/
double calc_dtconda(double b, struct cosmology *cosm)
{
  double a, dtconda, etab;
  a = b * b;
  etab = sqrt(cosm->omegam + cosm->omegav * a * a * a + cosm->omegak * a);
  dtconda = 2.0e0 / etab;
  return dtconda;
}

/*
  Evaluate H0*conformal time for FLRW cosmology.
  Assume Omega's passed in common to dtconda.
*/
double calc_tcon(double a, struct cosmology *cosm)
{
  double tcon;
  double b;
  b = sqrt(a);
  tcon = calc_rombin(calc_dtconda, 0.0e0, b, 1.0e-8, cosm);
  return tcon;
}

/*
 This function calculates the integrand for the normalization of the
 power spectrum with Delta = 1 at r = 8 Mpc/h.
*/
double calc_dsig8(double ak, struct cosmology *cosm)
{
  double dsig8 = 0.0;

  if(ak <= 0.0) return dsig8;

  /* Window function for spherical tophat of radius 8 Mpc/h. */
  double x = ak * 800.0 / cosm->h0;
  double w = 3.0 * (sin(x) - x * cos(x)) / (x * x * x);
  dsig8 = ak * ak * calc_pk_tot(ak, 1.0, cosm) * w * w;
  // dsig8=ak*ak*p(real(ak),1.0)*w*w;
  return dsig8;
}

/*
  Rombint returns the integral from a to b of f(x)dx using Romberg integration.
  The method converges provided that f(x) is continuous in (a,b).  The function
  f must be double precision and must be declared external in the calling
  routine.  tol indicates the desired relative accuracy in the integral.
  func=f, int_min=a , int_max=b
*/
double calc_rombin(double (*func)(double, struct cosmology *), double int_min, double int_max, double tol,
                   struct cosmology *cosm)
{
  int maxiter = 18, maxj = 5;
  double g[maxj + 1];

  double h = 0.5 * (int_max - int_min);
  double gmax = h * ((*func)(int_min, cosm) + (*func)(int_max, cosm));
  g[0] = gmax;
  int nint = 1;
  double error = 1.0e20;
  double rombin;

  double g0 = 0.0;

  for(int i = 0; i < maxiter; i++) {

    g0 = 0.0;
    for(int k = 0; k < nint; k++) {
      g0 += (*func)(int_min + (k + k + 1) * h, cosm);
    }

    g0 = 0.5 * g[0] + h * g0;
    h = 0.5 * h;
    nint = nint + nint;
    int jmax = fmin(i, maxj);
    double fourj = 1.0;

    for(int j = 0; j < jmax; j++) {
      /* Use Richardson extrapolation. */
      fourj = 4.0 * fourj;
      double g1 = g0 + (g0 - g[j]) / (fourj - 1.0);
      g[j] = g0;
      g0 = g1;
    }

    if(fabs(g0) >= tol) {
      error = 1.0 - gmax / g0;
    } else {
      error = gmax;
    }

    if((i >= (maxiter / 2) && fabs(error) <= tol)) break;

    gmax = g0;
    g[jmax] = g0;
  }

  rombin = g0;
  return rombin;
}

/*
  Evaluate the ultra spherical Bessel function j_2(k,chi,omegak), the
  generalization of spherical Bessel function to a constant curvature
  3-space.  Must have ak in units of H0/c and chi in units of c/H0.
*/
double calc_uj2(double ak, double chi, double omegak)
{
  double uj2;

  if(omegak < -1.0e-6) {
    fprintf(stderr, "Closed universe prohibited in uj2!\n");
    fprintf(stderr, "omegak = %g\n", omegak);
    exit(EXIT_FAILURE);
  } else if(omegak < 1.0e-6) {
    uj2 = calc_aj2(ak * chi);

  } else {
    double rc = 1.0 / sqrt(omegak);
    uj2 = calc_bj2(ak * rc, chi / rc);
  }

  return uj2;
}

/*  Evaluate the spherical bessel function j_2(x). */
double calc_aj2(double x)
{
  const double tol = 1.0e-16;
  double xa = fabs(x);
  double xx = x * x;

  double aj0, aj1, aj2;

  if(xa < 1.0e-3) {
    aj0 = 1.0 - xx / 6.0 * (1.0 - xx / 20.0 * (1.0 - xx / 42.0 * (1.0 - xx / 72.0)));
    aj1 = (x / 3.0) * (1.0 - xx / 10.0 * (1.0 - xx / 28.0 * (1.0 - xx / 54.0)));
  } else {
    aj0 = sin(x) / x;
    aj1 = (sin(x) - x * cos(x)) / xx;
  }

  /* Use power series expansion for small argument. */
  double x2 = -0.25 * xx;

  if(-x2 < 0.5 || xa < 1.5) {
    double fact = xx / 15.0;
    double sum = 1.0;
    double term = 1.0;
    int n = 0;

    while(n < 10 || fabs(term) > tol) {
      n = n + 1;
      term = term * x2 / (n * (n + 2.5));
      sum += term;
      if(fabs(sum) < 1.0e-6) break;
    }

    aj2 = fact * sum;
    return aj2;
  }

  /* Use recurrence relation to get aj2. */
  aj2 = 3.0 * aj1 / x - aj0;

  return aj2;
}

/*
 Evaluate the ultra spherical bessel function j_2(k,chi).
 This is the generalization of the spherical Bessel function to a pseudo
 3-sphere (a 3-space of constant negative curvature).  Must have the radial
 wavenumber k and coordinate chi be in units of the curvature distance,
 sqrt(-1/K).
*/
double calc_bj2(double ak, double chi)
{
  const double tol = 1.0e-10;
  double bj2, uj2;

  if(ak < 0.0 || chi < 0.0) {
    fprintf(stderr, "Negative ak, chi prohibited in bj2!\n");
    exit(EXIT_FAILURE);
  }

  double akk = ak * ak;
  double ak1 = sqrt(1.0 + akk);

  if(chi > 100.0) {
    fprintf(stderr, "r/Rcurv is too large in bj2\n");
    exit(EXIT_FAILURE);
  }

  double e = exp(chi);
  double ei = 1.0 / e;
  double ch = 0.5 * (e + ei);
  double sh = 0.5 * (e - ei);
  double ch2 = 0.5 * (1.0 + ch);
  double x2 = 0.5 * (1.0 - ch);
  double chi2 = chi * chi;

  if(sh < 1.0e-3) {
    sh = chi * (1.0 + chi2 / 6.0 * (1.0 + chi2 / 20.0 * (1.0 + chi2 / 42.0 * (1.0 + chi2 / 72.0))));

    x2 = -0.25 * chi2 * (1.0 + chi2 / 12.0 * (1.0 + chi2 / 30.0 * (1.0 + chi2 / 56.0 * (1.0 + chi2 / 90.0))));
  }

  const double tiny = 1.0e-30;

  double cth = ch / (sh + tiny);
  double cn = cos(ak * chi);
  double sn = sin(ak * chi);

  double uj0, uj1;

  if(sh > -tiny && sh < tiny) {
    uj0 = 1.0;
    uj1 = 0.0;
  } else if(ak > -tiny && ak < tiny) {
    uj0 = chi / sh;
    uj1 = (chi * ch - sh) / (sh * sh);
  } else {
    uj0 = sn / (ak * sh);
    uj1 = (sn * ch - ak * sh * cn) / (ak * ak1 * sh * sh);
  }

  if(-x2 < 0.5 && ak * chi < 2.0) {
    /* Use hypergeometric series expansion for small argument. */

    double fact = sh * ak1 / (3.0 * ch2 * sqrt(ch2));
    double ak2 = sqrt(akk + 4.0);
    fact = fact * sh * ak2 / (5.0 * ch2);
    double sum = 1.0;
    double term = 1.0;
    double n = 0.0;

    while(n < 8.0 || fabs(term) > tol) {
      n = n + 1.0;
      double hn = n - 0.5;
      term = term * x2 * (akk + hn * hn) / (n * (hn + 3));
      sum += term;
    }

    uj2 = fact * sum;

  } else {
    /* Use recurrence relation to get uj2. */
    if(sh > -tiny && sh < tiny) {
      uj2 = 0.0;
    } else {
      double ak2 = sqrt(akk + 4.0);
      uj2 = (3.0 * cth * uj1 - ak1 * uj0) / ak2;
    }
  }

  bj2 = uj2;

  return bj2;
}
