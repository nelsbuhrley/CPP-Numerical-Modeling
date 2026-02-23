# ==============================================================================
# Makefile for C/C++ + OpenMP on BYU Marylou Supercomputer
# Usage:
#   make          -> build optimized binary
#   make debug    -> build with debug flags
#   make release  -> build with maximum optimization (for production runs)
#   make clean    -> remove build artifacts
#   make info     -> print compiler and environment info
# ==============================================================================

# --- Compiler -----------------------------------------------------------------
CC        = gcc
CXX       = g++

# Adjust if your project is pure C or pure C++:
#   For .c files:   use $(CC) and CFLAGS
#   For .cpp files: use $(CXX) and CXXFLAGS

# --- Target binary ------------------------------------------------------------
TARGET    = my_program

# --- Source files -------------------------------------------------------------
# Automatically finds all .cpp and .c files in the current directory.
# Override manually if needed: SRCS = main.cpp utils.cpp
SRCS      = $(wildcard *.cpp) $(wildcard *.c)
OBJS      = $(SRCS:.cpp=.o)
OBJS     := $(OBJS:.c=.o)

# --- Flags --------------------------------------------------------------------
CFLAGS    = -Wall -Wextra -fopenmp -march=native -O2
CXXFLAGS  = -Wall -Wextra -fopenmp -march=native -O2 -std=c++17
LDFLAGS   = -fopenmp

# Debug flags (used with `make debug`)
DEBUG_FLAGS = -g -O0 -DDEBUG

# Release flags (used with `make release`) - maximum optimization
RELEASE_FLAGS = -O3 \
                -march=native \
                -mtune=native \
                -funroll-loops \
                -ffast-math \
                -fomit-frame-pointer \
                -finline-functions \
                -fprefetch-loop-arrays \
                -fvectorize \
                -ftree-vectorize \
                -ftree-loop-vectorize \
                -fopt-info-vec \
                -DNDEBUG

# --- Includes & Libraries -----------------------------------------------------
# Add any include paths or libraries you need:
# INCLUDES  = -I/path/to/include
# LIBS      = -L/path/to/lib -lsomething
INCLUDES  =
LIBS      =

# ==============================================================================
# Targets
# ==============================================================================

.PHONY: all debug release clean info

# Default: optimized build
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)
	@echo "Build complete: $(TARGET)"

# Compile .cpp files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile .c files
%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Debug build
debug: CFLAGS   += $(DEBUG_FLAGS)
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)
	@echo "Cleaned build artifacts."

# Print environment info (useful on the cluster to verify modules)
info:
	@echo "CC:      $(CC)  -> $$(which $(CC))"
	@echo "CXX:     $(CXX) -> $$(which $(CXX))"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "Sources:  $(SRCS)"
	@echo "OMP_NUM_THREADS: $${OMP_NUM_THREADS:-not set}"