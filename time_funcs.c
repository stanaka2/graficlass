#include <math.h>
#include "graficlass.h"
#include "prototype.h"

/*
  Evaluate a(tau) for Friedmann-Lemaitre-Robertson-Walker (FLRW) cosmology.
  (Use tauu here to avoid conflict with function tau.)
  Omegam := Omega today (a=1) in matter.
  Omegav := Omega today (a=1) in vacuum energy.
  dtau := H0*dt/a^2.
*/
double calc_aexp(double tauu, struct cosmology *cosm)
{
  /* Initial guess: matter-dominated solution. */
  double aexp = 1.0 / (1.0 - tauu * (sqrt(cosm->omegam) - 0.25 * cosm->omegam * tauu));

  /* Newton-Raphson iteration. */
  int niter = 0;
  double min_val;

  do {
    double eta = sqrt(cosm->omegam / aexp + cosm->omegav * aexp * aexp + cosm->omegak);

    double error = (calc_tau(aexp, cosm) - tauu) * aexp * aexp * eta;
    double aerr = fabs(error);
    double rerr = aerr / aexp;
    aexp = aexp - error;

    min_val = fmin(aerr, rerr);
    niter++;
  } while(min_val > 1.0e-6 && niter < 10);

  return aexp;
}

/*
  Evaluate tau(a) (inverse of a(tau)) for FLRW cosmology.
  Omegam := Omega today (a=1) in matter.
  Omegav := Omega today (a=1) in vacuum energy.
  dtau := H0*dt/a^2.
*/
double calc_tau(double a, struct cosmology *cosm)
{
  double adp = a;
  double tau = calc_rombin(calc_dtauda, 1.0e0, adp, 1.0e-8, cosm);
  return tau;
}

double calc_dtauda(double a, struct cosmology *cosm)
{
  double eta = sqrt(cosm->omegam / a + cosm->omegav * a * a + cosm->omegak);
  double dtauda = 1.0e0 / (a * a * eta);
  return dtauda;
}

/*
  Evaluate dln(a)/dtau for FLRW cosmology.
  Omegam := Omega today (a=1) in matter.
  Omegav := Omega today (a=1) in vacuum energy.
*/
double calc_dladt(double a, struct cosmology *cosm)
{
  double dladt, eta;
  eta = sqrt(cosm->omegam / a + cosm->omegav * a * a + cosm->omegak);

  /* N.B. eta=a*H/H0, dladt = da/(H0*dtau) where tau is conformal time! */
  dladt = a * eta;
  return dladt;
}

/*
  Evaluate D+(a) (linear growth factor) for FLRW cosmology.
  Omegam := Omega today (a=1) in matter.
  Omegav := Omega today (a=1) in vacuum energy.
*/
double calc_dplus(double a, struct cosmology *cosm)
{
  double dplus, eta, adp;
  adp = a;
  eta = sqrt(cosm->omegam / a + cosm->omegav * a * a + cosm->omegak);
  dplus = eta / a * calc_rombin(calc_ddplus, 0.0, adp, 1.0e-8, cosm);
  return dplus;
}

double calc_ddplus(double a, struct cosmology *cosm)
{
  double ddplus = 0.0;

  // a==0.0
  if(a > -1.0e-10 && a < 1.0e-10) return ddplus;

  double eta = sqrt(cosm->omegam / a + cosm->omegav * a * a + cosm->omegak);
  ddplus = 2.5e0 / (eta * eta * eta);
  return ddplus;
}

/* Inverts the function dpls=dplus(a,omegam,omegav) for a=adp. */
double calc_adp(double dpls, struct cosmology *cosm)
{
  double adp = 0.0;

  // adp==0.0
  if(dpls > -1.0e-10 && dpls < 1.0e-10) return adp;

  /* Initial guess. */
  adp = 1.0e-3;

  /* Newton-Raphson iteration. */
  int niter = 0;
  double min_val;

  do {
    double dpls0 = calc_dplus(adp, cosm);
    double ddplda = dpls0 * calc_fomega(adp, cosm) / adp;

    double error = (dpls0 - dpls) / ddplda;
    double aerr = fabs(error);
    double rerr = aerr / adp;
    adp = adp - error;

    min_val = fmin(aerr, rerr);

    niter++;
  } while(min_val > 1.0e-6 && niter < 10);

  return adp;
}

/*
 Evaluate f := dlog[D+]/dlog[a] (logarithmic linear growth rate) for
 lambda+matter-dominated cosmology.
 Omega0 := Omega today (a=1) in matter only.  Omega_lambda = 1 - Omega0.
*/
double calc_fomega(double a, struct cosmology *cosm)
{
  double fomega = 1.0;
  const double tiny = 1.0e-10;

  // omegam==1.0 and omegav==0.0
  if((cosm->omegav > -tiny && cosm->omegav < tiny) && (cosm->omegam > 1.0 - tiny && cosm->omegam < 1.0 + tiny))
    return fomega;

  double eta = sqrt(cosm->omegam / a + cosm->omegav * a * a + cosm->omegak);
  fomega = (2.5 / calc_dplus(a, cosm) - 1.5 * cosm->omegam / a - cosm->omegak) / (eta * eta);
  return fomega;
}

double calc_fomega_2LPT(double a, struct cosmology *cosm)
{
  const double tiny = 1.0e-5;

  double normalized_H2 = cosm->omegam / CUBE(a) + cosm->omegak / SQR(a) + cosm->omegav;
  // time dependent Omega_matter
  double omg_m = cosm->omegam / CUBE(a) / normalized_H2;

  double f2;

  if(cosm->omegav > -tiny && cosm->omegav < tiny) {
    f2 = 2.0 * pow(omg_m, 4.0 / 7.0);
  } else if(cosm->omegam + cosm->omegav + cosm->omegar > 1.0 - tiny &&
            cosm->omegam + cosm->omegav + cosm->omegar < 1.0 + tiny) {
    f2 = 2.0 * pow(omg_m, 6.0 / 11.0);
  } else {
    fprintf(stderr, "Invalid cosmology in calc_fomega_2LPT\n");
    exit(EXIT_FAILURE);
  }

  return f2;
}

double calc_dplus_2LPT(double a, struct cosmology *cosm)
{
  double Dplus_2LPT;
  const double tiny = 1.0e-5;

  double Dplus = calc_dplus(a, cosm);

  double normalized_H2 = cosm->omegam / CUBE(a) + cosm->omegak / SQR(a) + cosm->omegav;
  // time dependent Omega_matter
  double omg_m = cosm->omegam / CUBE(a) / normalized_H2;

  if(cosm->omegav > -tiny && cosm->omegav < tiny) {
    Dplus_2LPT = -3.0 / 7.0 * SQR(Dplus) * pow(omg_m, -2.0 / 63.0);
  } else if(cosm->omegam + cosm->omegav + cosm->omegar > 1.0 - tiny &&
            cosm->omegam + cosm->omegav + cosm->omegar < 1.0 + tiny) {
    Dplus_2LPT = -3.0 / 7.0 * SQR(Dplus) * pow(omg_m, -1.0 / 143.0);
  } else {
    fprintf(stderr, "Invalid cosmology in calc_dplus_2LPT\n");
    exit(EXIT_FAILURE);
  }

  return Dplus_2LPT;
}
