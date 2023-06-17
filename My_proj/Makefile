CC = mpic++
COPTS = -g -std=c++11 -fPIC -O0 -w -Wstrict-aliasing -fstrict-aliasing

SRC_DIR := src
PAR_DIR := parmetis
OBJ_DIR := obj
BIN_DIR := .
EXEBLOCK := blocktest
SRC := $(wildcard $(SRC_DIR)/*.cpp)
EXEPAR := parmetistest
PAR := $(wildcard $(PAR_DIR)/*.cpp)

OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
PAR_OBJ := $(OBJ_DIR)/parmetistest.o $(OBJ_DIR)/utilities.o $(OBJ_DIR)/matrices.o

CPPFLAGS := -Iinclude -MMD -MP # -I is a preprocessor flag, not a compiler flag
LDFLAGS  := -Llib              # -L is a linker flag
LDLIBS   := -lmetis -lGKlib -lparmetis              

all: $(EXEBLOCK) $(EXEPAR)

.PHONY: all clean

$(EXEBLOCK): $(OBJ) | $(BIN_DIR)
	$(CC) $(COPTS) $(CPPFLAGS) -o $@ $^

$(EXEPAR): parmetistest.o 
	$(CC) $(COPTS) $(LDFLAGS) -o $@ $(PAR_OBJ) $(LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(COPTS) $(CPPFLAGS) -c $< -o $@

parmetistest.o: 
	$(CC) $(COPTS) $(CPPFLAGS) -c $(PAR_DIR)/parmetistest.cpp -o $(OBJ_DIR)/$@ $(LDLIBS)

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	@$(RM) -rv $(BIN_DIR)/$(EXEBLOCK) $(BIN_DIR)/$(EXEPAR) $(OBJ_DIR) # The @ disables the echoing of the command

-include $(OBJ:.o=.d) # The dash silences errors when files don't exist (yet)