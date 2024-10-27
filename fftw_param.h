#pragma once

#include <mpi.h>
#include <fftw3-mpi.h>

struct fftw_param {
  int mpi_rank, mpi_nproc;

  ptrdiff_t total_fft_mesh[3];
  ptrdiff_t local_rsize, local_csize;
  ptrdiff_t local_rlength[3], local_rstart[3];
  ptrdiff_t local_clength[3], local_cstart[3];
};
