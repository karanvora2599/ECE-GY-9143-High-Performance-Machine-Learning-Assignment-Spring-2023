### ECE-GY 9143 - High Performance Computing for Machine Learning
### Instructor: Parijat Dubey
### Makefile for Part A Assignment 5

SDK_INSTALL_PATH :=  /usr/local/cuda
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := YourNameCUDA1.tar
EXECS :=  vecadd01 matmult01
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.o

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

timer.o : timer.cu timer.h
	${NVCC} $< -c -o $@ $(OPTIONS)

#######################################################################
# vecaddKernel00.o : vecaddKernel00.cu
# 	${NVCC} $< -c -o $@ $(OPTIONS)

# vecadd00 : vecadd.cu vecaddKernel.h vecaddKernel00.o timer.o
# 	${NVCC} $< vecaddKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
vecaddKernel01.o : vecaddKernel01.cu
	${NVCC} $< -c -o $@ $(OPTIONS)

vecadd01 : vecadd.cu vecaddKernel.h vecaddKernel01.o timer.o
	${NVCC} $< vecaddKernel01.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
## Provided Kernel
# matmultKernel00.o : matmultKernel00.cu matmultKernel.h 
# 	${NVCC} $< -c -o $@ $(OPTIONS)

# matmult00 : matmult.cu  matmultKernel.h matmultKernel00.o timer.o
# 	${NVCC} $< matmultKernel00.o -o $@ $(LIB) timer.o $(OPTIONS)


#######################################################################
## Expanded Kernel, notice that FOOTPRINT_SIZE is redefined (from 16 to 32)
matmultKernel01.o : matmultKernel01.cu matmultKernel.h
	${NVCC} $< -c -o $@ $(OPTIONS) -DFOOTPRINT_SIZE=32

matmult01 : matmult.cu  matmultKernel.h matmultKernel01.o timer.o
	${NVCC} $< matmultKernel01.o -o $@ $(LIB) timer.o $(OPTIONS) -DFOOTPRINT_SIZE=32




