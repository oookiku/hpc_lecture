# GNU Makefile
# @author Daisuke Kikuta
# @date   04/21/2020 (Final update)
# @server TSUBAME3.0 

#-----------------------------------------------------------------------------+
#                           PATH & Option Setting                             |
#-----------------------------------------------------------------------------+
# CUDA
NVCC      := nvcc
NVCCFLAGS := -arch=sm_60 -std=c++14 -O3 #-g -G -O0
CUDA_PATH := /apps/t3/sles12sp2/cuda/10.2.89
INCLUDE   += -I$(CUDA_PATH)/include --device-c
LDFLAGS   += -L$(CUDA_PATH)/lib64 -lcudart

#-----------------------------------------------------------------------------+
#                            Rules for building codes                         |
#-----------------------------------------------------------------------------+
PROJECT   := main
CXXFILES  := $(wildcard *.cpp)
OBJECT    := $(foreach src, $(CXXFILES), $(patsubst %.cpp, %.o, $(src)))
NVCCFILES += $(wildcard *.cu)  
OBJECT    += $(foreach src, $(NVCCFILES), $(patsubst %.cu, %.o, $(src)))

# Suffixes rule
.SUFFIXES: .cu .o
.cu.o:
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< 

# Linker
$(PROJECT) : $(OBJECT)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@ 

#-----------------------------------------------------------------------------+
#                                 Cleaning                                    |
#-----------------------------------------------------------------------------+
.PHONY: clean
clean :
	$(RM) $(PROJECT) $(OBJECT)
