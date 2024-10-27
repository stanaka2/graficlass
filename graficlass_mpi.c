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
#include "fftw_param.h"
#include "constants.h"
#include "prototype.h"

void read_input_graficlassfile(char *, struct run_param *);
void set_params(struct run_param *);
void set_fftw_params(struct fftw_param *, struct run_param *);

int main(int argc, char **argv)
{
  if(argc != 2) {
    fprintf(stderr, "Usage :: mpiexec -n <node> %s <grafic param file>\n", argv[0]);
    exit(EXIT_FAILURE);
  }

#ifdef __OUTPUT_CDM_FOR_CB__
  printf("output CDM file is CDM + baryon components.\n");
#else
  printf("output CDM file is CDM component. Compile with __OUTPUT_CDM_FOR_CB__ macro, it will add "
         "baryon component.\n");
#endif

  struct run_param this_run;
  struct fftw_param this_fft;

  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  this_run.fft_thread_flag = provided >= MPI_THREAD_FUNNELED;

  MPI_Comm_size(MPI_COMM_WORLD, &(this_run.mpi_nproc));
  MPI_Comm_rank(MPI_COMM_WORLD, &(this_run.mpi_rank));

  this_run.nx_tot = NPX;
  this_run.ny_tot = NPY;
  this_run.nz_tot = NPZ;
  this_run.np_tot = this_run.nx_tot * this_run.ny_tot * this_run.nz_tot;

  if(this_run.mpi_rank == 0)
    printf("(NPX, NPY, NPZ) = (%lu, %lu, %lu) ; NP_TOT = %lu\n", this_run.nx_tot, this_run.ny_tot, this_run.nz_tot,
           this_run.np_tot);

  this_run.sigstart = 0.1;

  static char inputfile[256];
  sprintf(inputfile, "%s", argv[1]);

  read_input_graficlassfile(inputfile, &this_run);

  pini(&this_run);
  set_params(&this_run);

  MPI_Barrier(MPI_COMM_WORLD);

  set_fftw_params(&this_fft, &this_run);

  this_run.nx_loc = this_fft.local_rlength[0];
  this_run.ny_loc = this_fft.local_rlength[1];
  this_run.nz_loc = this_fft.local_rlength[2];

  this_run.nx_loc_start = this_fft.local_rstart[0];
  this_run.ny_loc_start = this_fft.local_rstart[1];
  this_run.nz_loc_start = this_fft.local_rstart[2];

  this_run.np_loc = this_run.nx_loc * this_run.ny_loc * this_run.nz_loc;

  double safe_factor = 1.1;
  this_run.fft_local_rsize = this_fft.local_rsize * safe_factor;

  printf("rank %d : (LOC_NPX, LOC_NPY, LOC_NPZ) = (%lu, %lu, %lu) ; (STL_NPX, "
         "STL_NPY, STL_NPZ) = (%lu, %lu, %lu) ; NP_LOC = %lu, FFT_LOC = %lu\n",
         this_run.mpi_rank, this_fft.local_rlength[0], this_fft.local_rlength[1], this_fft.local_rlength[2],
         this_fft.local_rstart[0], this_fft.local_rstart[1], this_fft.local_rstart[2], this_run.np_loc,
         this_run.fft_local_rsize);

  this_run.npadd = 1;

  output_cdm_data_mpi(&this_run);
  // output_baryonic_data_mpi(&this_run);

  if(this_run.nu_mass_tot > 1.0e-3) {
    for(int inu = 0; inu < this_run.mass_nu_num; inu++) {
      output_neutrino_data_mpi(&this_run, inu);
    }
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}
