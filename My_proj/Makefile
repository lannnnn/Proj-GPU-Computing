CC = mpic++
COPTS = -g -std=c++11 -fPIC -O0 -Wall -Wstrict-aliasing -fstrict-aliasing

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := .

EXE := blocktest
SRC := $(wildcard $(SRC_DIR)/*.cpp)

OBJ := $(SRC:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

CPPFLAGS := -Iinclude -MMD -MP # -I is a preprocessor flag, not a compiler flag
CFLAGS   := -Wall              # some warnings about bad code
LDFLAGS  := -Llib              # -L is a linker flag
LDLIBS   := -lmetis -lGKlib -lparmetis              

all: $(EXE)

.PHONY: all clean

$(EXE): $(OBJ) | $(BIN_DIR)
	$(CC) $(COPTS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CC) $(COPTS) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

clean:
	@$(RM) -rv $(BIN_DIR) $(OBJ_DIR) # The @ disables the echoing of the command

-include $(OBJ:.o=.d) # The dash silences errors when files don't exist (yet)