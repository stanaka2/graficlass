#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*
 Splder fits a cubic spline to y and returns the first derivatives at
 the grid points in dy.  dy is equivalent to a 4th-order Pade
 difference formula for dy/di.
*/

#define NSPL (100001)

void fit_splder(double *y, double *dy, double *gspl, int n)
{
  int n1 = n - 1;

  double *f;
  f = (double *)malloc(sizeof(double) * n);

  /* Quartic fit to dy/di at boundaries, assuming d3y/di3=0. */
  f[0] = (-10.0 * y[0] + 15.0 * y[1] - 6.0 * y[3] + y[4]) / 6.0;
  f[n1] = (10.0 * y[n1] - 15.0 * y[n1 - 1] + 6.0 * y[n1 - 2] - y[n1 - 3]) / 6.0;

  /*
    Solve the tridiagonal system
    dy(i-1)+4*dy(i)+dy(i+1)=3*(y(i+1)-y(i-1)), i=2,3,...,n1,
    with dy(1)=f(1), dy(n)=f(n).
  */

  for(int i = 1; i < n1; i++) {
    f[i] = gspl[i] * (3.0 * (y[i + 1] - y[i - 1]) - f[i - 1]);
  }

  dy[n1] = f[n1];
  for(int i = n1 - 1l; i >= 0; i--) {
    dy[i] = f[i] - gspl[i] * dy[i + 1];
  }

  free(f);
}

/*
  Splini must be called before splder to initialize array g in common.
 */
void splini(double *gspl, int n)
{
  assert(n == NSPL);

  gspl[0] = 0.0;

  for(int i = 1; i < n; i++) {
    gspl[i] = 1.0 / (4.0 - gspl[i - 1]);
  }
}
