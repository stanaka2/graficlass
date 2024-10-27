CC = gcc
MPICC = mpicc
F77 = gfortran
OMP_FLAG=-fopenmp
CFLAGS = -O3 -fopenmp -std=c11 -fPIE

CFLAGS += -D__OUTPUT_CDM_FOR_CB__

CFLAGS += -D__2LPT__
#CFLAGS += -D__DEBUG__

#CC=icc
#F77=ifort
#FFLAGS=-ffixed-line-length-132

FFTW_DIR=
#FFTW_DIR=/usr/local/fftw-3

ifeq ($(site),fugaku)
CC=fccpx
MPICC=mpifccpx
CFLAGS = -O3 -Kopenmp -std=c11
FFTW_DIR=/vol0006/hp120286/data/u10067/software/build/fftw/3.3.10
endif

ifeq ($(FFTW_DIR),)
FFTWLIBS =  -lfftw3f_omp -lfftw3f
FFTWMPILIBS = -lfftw3f_mpi -lfftw3f_omp -lfftw3f
else
FFTWLIBS = -L$(FFTW_DIR)/lib -lfftw3f_omp -lfftw3f
FFTWMPILIBS = -L$(FFTW_DIR)/lib -lfftw3f_mpi  -lfftw3f_omp  -lfftw3f
CFLAGS += -I$(FFTW_DIR)/include
endif

GRAFICLASS_OBJ = input_graficlass_params.o power_funcs.o time_funcs.o util_funcs.o transfer_table.o system_call.o
G1OBJ = graficlass.o  ic4.o  output_ic_data.o $(GRAFICLASS_OBJ)
G1MPIOBJ = graficlass_mpi.o  ic4_mpi.o  output_ic_data_mpi.o set_fftw_param.o $(GRAFICLASS_OBJ)

CHECKGRAFICOBJ = check_grafic_data.o

all: graficlass_mpi

.c.o:
	$(MPICC) $(CFLAGS) -c $<

.f.o:
	$(F77) $(FFLAGS) -c $<

.F.o:
	$(F77) -c $<


graficlass: $(G1OBJ)
	$(CC) $(CFLAGS) $(G1OBJ) $(FFTWLIBS) -o $@ -lm

graficlass_mpi: $(G1MPIOBJ)
	$(MPICC) $(CFLAGS) $(G1MPIOBJ) $(FFTWMPILIBS) -o $@ -lm

check_grafic_data : $(CHECKGRAFICOBJ)
	$(CC) $(CFLAGS) $(CHECKGRAFICOBJ) -o $@


clean:
	rm -rf *.o graficlass graficlass_mpi  check_grafic_data

distclean: clean
	rm -rf *~
	rm -f ic_*.dat
