/*
  Generate initial conditions for cosmological N-body integration
  as a Gaussian random field.
  This version does not do constraints.  Instead, it produces output
  compatible with grafic2 for multiscale initial conditions.
  Disk use: lun=10 is output file, lun=11 is temp files.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "graficlass.h"
#include "constants.h"
#include "prototype.h"

void read_input_graficlassfile(char *, struct run_param *);
void set_params(struct run_param *);

int main(int argc, char **argv)
{
  if(argc != 2) {
    fprintf(stderr, "Usage :: %s <grafic param file>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

#ifdef __OUTPUT_CDM_FOR_CB__
  printf("output CDM file is CDM + baryon components.\n");
#else
  printf("output CDM file is CDM component. Compile with __OUTPUT_CDM_FOR_CB__ macro, it will add "
         "baryon component.\n");
#endif

  struct run_param this_run;

  this_run.mpi_rank == 0;

  this_run.nx_tot = NPX;
  this_run.ny_tot = NPY;
  this_run.nz_tot = NPZ;
  this_run.np_tot = this_run.nx_tot * this_run.ny_tot * this_run.nz_tot;

  printf("(NPX, NPY, NPZ) = (%lu, %lu, %lu) ; NP_TOT = %lu\n", this_run.nx_tot, this_run.ny_tot, this_run.nz_tot,
         this_run.np_tot);

  this_run.sigstart = 0.1;

  static char inputfile[256];
  sprintf(inputfile, "%s", argv[1]);

  read_input_graficlassfile(inputfile, &this_run);

  pini(&this_run);
  set_params(&this_run);

  output_cdm_data(&this_run);
  // output_baryonic_data(&this_run);

  if(this_run.nu_mass_tot > 1.0e-3) {
    for(int inu = 0; inu < this_run.mass_nu_num; inu++) {
      output_neutrino_data(&this_run, inu);
    }
  }

  return EXIT_SUCCESS;
}
