CC = g++
MPICC = mpic++
COPTS = -g -std=c++11 -fPIC -O0 -w -Wstrict-aliasing -fstrict-aliasing

CUDA_PATH ?= $(CUDA_HOME)
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

SRC_DIR := src
PAR_DIR := parmetis
OBJ_DIR := obj
BIN_DIR := .
EXEBLOCK := blocktest
SRC := $(wildcard $(SRC_DIR)/*.cpp)
EXEPAR := parmetistest
PAR := $(wildcard $(PAR_DIR)/*.cpp)
EXECUDA := block_cuda
EXEREFCUDA := ref_block_cuda

OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
PAR_OBJ := $(OBJ_DIR)/parmetistest.o $(OBJ_DIR)/utilities.o $(OBJ_DIR)/csr.o $(OBJ_DIR)/group.o

CPPFLAGS := -Iinclude -MMD -MP # -I is a preprocessor flag, not a compiler flag
LDFLAGS  := -Llib              # -L is a linker flag
LDLIBS   := -lmetis -lGKlib -lparmetis          
CUDA_INCLUDE  = -I include/cuda -I $(CUDA_PATH)/include 

all: $(EXEBLOCK) 
par: $(EXEPAR)

$(EXEBLOCK): $(OBJ) | $(BIN_DIR)
	$(CC) $(COPTS) $(CPPFLAGS) -o $@ $^ $(LDLIBS)

$(EXEPAR): parmetistest.o 
	$(MPICC) $(COPTS) $(LDFLAGS) -o $@ $(PAR_OBJ) $(LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(COPTS) $(CPPFLAGS) -c $< -o $@

parmetistest.o: 
	$(MPICC) $(COPTS) $(CPPFLAGS) -c $(PAR_DIR)/parmetistest.cpp -o $(OBJ_DIR)/$@ $(LDLIBS)

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

#-------------------------------------------------------------------------
#CUDA PROGRAMS
CUDA_OBJ_DIR = $(OBJ_DIR)/cuda
CUDA_SRC_DIR = $(SRC_DIR)/cuda

CUDA_OBJECTS := $(OBJ_DIR)/utilities.o $(OBJ_DIR)/csr.o $(OBJ_DIR)/group.o 

CUDA_CXXFLAGS += -std=c++11 -m64 -gencode=arch=compute_80,code=sm_80 -w -g -G

$(CUDA_OBJ_DIR)/block_cuda.o: $(CUDA_SRC_DIR)/block_cuda.cu
	mkdir -p $(CUDA_OBJ_DIR)
	$(NVCC) $(CUDA_CXXFLAGS) $(CPPFLAGS) $(CUDA_INCLUDE) -c $< -o $@

$(CUDA_OBJ_DIR)/ref_block_cuda.o: $(CUDA_SRC_DIR)/ref_block_cuda.cu
	mkdir -p $(CUDA_OBJ_DIR)
	$(NVCC) $(CUDA_CXXFLAGS) $(CPPFLAGS) $(CUDA_INCLUDE) -c $< -o $@ 

$(CUDA_OBJ_DIR)/group_cuda.o: $(CUDA_SRC_DIR)/group_cuda.cu
	mkdir -p $(CUDA_OBJ_DIR)
	$(NVCC) $(CUDA_CXXFLAGS) $(CPPFLAGS) $(CUDA_INCLUDE) -c $< -o $@	

$(BIN_DIR)/$(EXECUDA) : $(CUDA_OBJ_DIR)/block_cuda.o $(CUDA_OBJ_DIR)/group_cuda.o $(CUDA_OBJECTS)  
	$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_INCLUDE) -o $@ $(CUDA_OBJ_DIR)/block_cuda.o $(CUDA_OBJ_DIR)/group_cuda.o $(CUDA_OBJECTS) 

$(BIN_DIR)/$(EXEREFCUDA) : $(CUDA_OBJ_DIR)/ref_block_cuda.o $(CUDA_OBJ_DIR)/group_cuda.o $(CUDA_OBJECTS)  
	$(NVCC) $(CUDA_CXXFLAGS) $(CUDA_INCLUDE) -o $@ $(CUDA_OBJ_DIR)/ref_block_cuda.o $(CUDA_OBJ_DIR)/group_cuda.o $(CUDA_OBJECTS) 

#-------------------------------------------------------------------------

.PHONY: all clean

cuda :   $(BIN_DIR)/$(EXEREFCUDA)

clean:
	@$(RM) -rv $(BIN_DIR)/$(EXEBLOCK) $(BIN_DIR)/$(EXEPAR) $(BIN_DIR)/$(EXECUDA) $(BIN_DIR)/$(EXEREFCUDA) $(OBJ_DIR) # The @ disables the echoing of the command

-include $(OBJ:.o=.d) # The dash silences errors when files don't exist (yet)