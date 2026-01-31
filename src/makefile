# Makefile for TopOpt_in_PETSc with GPU support
# Updated for PETSc 3.21.0 with CUDA

PETSC_DIR=/home/user/petsc-cuda/petsc-3.21.0
PETSC_ARCH=arch-cuda-mpi
CFLAGS = -I.
FFLAGS=
CPPFLAGS=-I.
FPPFLAGS=
LOCDIR=
EXAMPLESC=
EXAMPLESF=
MANSEC=
CLEANFILES=
NP=

# CUDA编译器
NVCC = nvcc
NVCCFLAGS = -O3 -arch=sm_80 -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include -I/home/user/openmpi-cuda/include -I/usr/local/cuda-12.2/include

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

# CUDA对象文件
MatrixFreeGPU.o: MatrixFreeGPU.cu MatrixFreeGPU.h
	${NVCC} ${NVCCFLAGS} -c MatrixFreeGPU.cu -o MatrixFreeGPU.o

topopt: main.o TopOpt.o LinearElasticity.o MMA.o OC.o Filter.o PDEFilter.o MPIIO.o MatrixFreeGPU.o
	-${CLINKER} -o topopt main.o TopOpt.o LinearElasticity.o MMA.o OC.o Filter.o PDEFilter.o MPIIO.o MatrixFreeGPU.o ${PETSC_LIB} -lcudart -L/usr/local/cuda-12.2/lib64

myclean:
	rm -rf topopt *.o output* binary* log* makevtu.pyc Restart* 
