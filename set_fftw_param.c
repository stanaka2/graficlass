#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>

#include "graficlass.h"
#include "fftw_param.h"

/* a/b */
static int ceili(int a, int b) { return (a + b - 1) / b; }

ptrdiff_t get_fft_mpi_local_length_1d(ptrdiff_t ntot, int nproc, int irank, ptrdiff_t *length, ptrdiff_t *start)
{
  ptrdiff_t base_length = ceili(ntot, nproc);
  ptrdiff_t base_start = irank * (base_length);

  ptrdiff_t ret_base_length = base_length;

  if(base_length + base_start >= ntot) {

    base_length = ntot - base_start;

    if(base_length <= 0) {
      base_length = 0;
      base_start = 0;
    }
  }

  *length = base_length;
  *start = base_start;

  return ret_base_length;
}

void set_fftw_params(struct fftw_param *this_fft, struct run_param *this_run)
{
  this_fft->mpi_nproc = this_run->mpi_nproc;
  this_fft->mpi_rank = this_run->mpi_rank;

  this_fft->total_fft_mesh[0] = this_run->nx_tot;
  this_fft->total_fft_mesh[1] = this_run->ny_tot;
  this_fft->total_fft_mesh[2] = this_run->nz_tot;

  if(this_run->fft_thread_flag) fftwf_init_threads();

  fftwf_mpi_init();

  this_fft->local_rsize =
      fftwf_mpi_local_size_3d(this_fft->total_fft_mesh[0], this_fft->total_fft_mesh[1], this_fft->total_fft_mesh[2] + 2,
                              MPI_COMM_WORLD, &(this_fft->local_rlength[0]), &(this_fft->local_rstart[0]));

  this_fft->local_clength[0] = this_fft->local_rlength[0];
  this_fft->local_cstart[0] = this_fft->local_rstart[0];

  this_fft->local_rlength[1] = this_fft->total_fft_mesh[1];
  this_fft->local_rlength[2] = this_fft->total_fft_mesh[2];

  this_fft->local_rstart[1] = 0;
  this_fft->local_rstart[2] = 0;

  this_fft->local_clength[1] = this_fft->local_rlength[1];
  this_fft->local_clength[2] = (this_fft->total_fft_mesh[2] + 2) / 2;

  this_fft->local_cstart[1] = 0;
  this_fft->local_cstart[2] = 0;

  this_fft->local_csize = this_fft->local_clength[0] * this_fft->local_clength[1] * this_fft->local_clength[2];

#if 0
  for(int irank=0; irank<this_fft->mpi_nproc; irank++) {
    if(this_fft->mpi_rank==irank) {
      printf("%d llength,start  %td %td %td , %td %td %td\n", irank,
	     this_fft->local_rlength[0], this_fft->local_rlength[1], this_fft->local_rlength[2],
	     this_fft->local_rstart[0], this_fft->local_rstart[1], this_fft->local_rstart[2]);
      printf("%d clength,start  %td %td %td , %td %td %td\n", irank,
	     this_fft->local_clength[0], this_fft->local_clength[1], this_fft->local_clength[2],
	     this_fft->local_cstart[0], this_fft->local_cstart[1], this_fft->local_cstart[2]);
      printf("%d lsize,csize  %td %td\n", irank, this_fft->local_rsize, this_fft->local_csize);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif

  ptrdiff_t tot_length = 0;
  this_run->mpi_rank_edge = this_run->mpi_nproc;

  if(this_run->mpi_rank == 0) {
    printf("local_size : ");
  }

  for(int irank = 0; irank < this_run->mpi_nproc; irank++) {
    ptrdiff_t length_x, start_x, base_length_x;
    base_length_x = get_fft_mpi_local_length_1d(this_run->nx_tot, this_run->mpi_nproc, irank, &length_x, &start_x);
    tot_length += length_x;

    if(length_x == 0 && this_run->mpi_rank_edge == this_run->mpi_nproc) {
      this_run->mpi_rank_edge = irank;
    }
    if(this_run->mpi_rank == 0) printf("%td ", length_x);
  }
  if(this_run->mpi_rank == 0) {
    printf("\n");
    printf("edge rank %d\n", this_run->mpi_rank_edge);
  }

  assert(tot_length == this_run->nx_tot);
}
#if 0
int main(int argc, char **argv)
{
  struct run_param this_run;
  struct fftw_param this_fft;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &(this_run.mpi_nproc));
  MPI_Comm_rank(MPI_COMM_WORLD, &(this_run.mpi_rank));

  uint64_t nx = 347;

  this_run.nx_tot = nx;
  this_run.ny_tot = nx;
  this_run.nz_tot = nx;
  this_run.np_tot = this_run.nx_tot*this_run.ny_tot*this_run.nz_tot;

  set_fftw_params(&this_fft, &this_run);

  return EXIT_SUCCESS;
}
#endif
