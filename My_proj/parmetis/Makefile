CC = g++
MPICC = mpic++
COPTS = -g -std=c++11 -fPIC -O0 -w -Wstrict-aliasing -fstrict-aliasing

CUDA_PATH ?= $(CUDA_HOME)
NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(CXX)

SRC_DIR := ../src
PAR_DIR := .
OBJ_DIR := ../obj
BIN_DIR := .
EXEPAR := parmetistest
EXECAL := calculate_dense
PAR := $(wildcard $(PAR_DIR)/*.cpp)

OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
PAR_OBJ := $(OBJ_DIR)/parmetistest.o $(OBJ_DIR)/utilities.o $(OBJ_DIR)/csr.o $(OBJ_DIR)/group.o
CAL_OBJ := $(OBJ_DIR)/calculate_dense.o $(OBJ_DIR)/utilities.o $(OBJ_DIR)/csr.o $(OBJ_DIR)/group.o 

METIS_INCLUDE = -I../include
METIS_LIB = -L../lib

CPPFLAGS := -Iinclude -MMD -MP # -I is a preprocessor flag, not a compiler flag
LDFLAGS  := -Llib              # -L is a linker flag
LDLIBS   := -lmetis -lGKlib -lparmetis
CUDA_INCLUDE  = -I include/cuda -I $(CUDA_PATH)/include 

all: $(EXEPAR) $(EXECAL)

$(EXEPAR): parmetistest.o 
	$(MPICC) $(COPTS) $(LDFLAGS) -o $@ $(PAR_OBJ) $(METIS_LIB) $(LDLIBS)

$(EXECAL): calculate_dense.o 
	$(MPICC) $(COPTS) $(LDFLAGS) -o $@ $(CAL_OBJ) $(METIS_LIB) $(LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(COPTS) $(CPPFLAGS) $(METIS_INCLUDE) -c $< -o $@

parmetistest.o: 
	$(MPICC) $(COPTS) $(CPPFLAGS) $(METIS_INCLUDE) -c $(PAR_DIR)/parmetistest.cpp -o $(OBJ_DIR)/$@ $(LDLIBS)

calculate_dense.o: 
	$(MPICC) $(COPTS) $(CPPFLAGS) $(METIS_INCLUDE) -c $(PAR_DIR)/calculate_dense.cpp -o $(OBJ_DIR)/$@ $(LDLIBS)

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

.PHONY: all clean

clean:
	@$(RM) -rv $(BIN_DIR)/$(EXEBLOCK) $(BIN_DIR)/$(EXEPAR) $(BIN_DIR)/$(EXECUDA) $(BIN_DIR)/$(EXEREFCUDA) $(OBJ_DIR) # The @ disables the echoing of the command

-include $(OBJ:.o=.d) # The dash silences errors when files don't exist (yet)