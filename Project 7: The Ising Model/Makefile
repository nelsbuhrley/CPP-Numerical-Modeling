CXX = g++
CXXFLAGS = -std=c++17 -Iinclude -I../include -I/usr/local/opt/libomp/include -Xpreprocessor -fopenmp
LDFLAGS = -L/usr/local/opt/libomp/lib -lomp -lz

DEBUG_FLAGS = -O0 -g \
	-Wall \
	-Wextra

RELEASE_FLAGS = -O3 -DNDEBUG \
	-march=native \
	-mtune=native \
	-flto \
	-funroll-loops \
	-ftree-vectorize \
	-fno-math-errno \
	-fomit-frame-pointer \
	-finline-functions \
	-fstrict-aliasing \
	-falign-functions=32 \
	-falign-loops=32

UNSAFE_RELEASE_FLAGS = -ffast-math


# Directories
INCLUDE_DIR = include
BIN_DIR = bin

# Sources and headers
SOURCES = main.cpp ../src/cnpy.cpp
HEADERS = $(wildcard $(INCLUDE_DIR)/*.h)

# Executable name
TARGET = $(BIN_DIR)/main

# Default target
all: $(TARGET)

# Build the program
$(TARGET): $(SOURCES) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES) $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: LDFLAGS := $(filter-out -lomp,$(LDFLAGS))
debug: clean $(TARGET)

# Release build
release: CXXFLAGS += $(RELEASE_FLAGS)
release: LDFLAGS += -flto
release: clean $(TARGET)

# Unsafe release build (with aggressive optimizations)
unsafe: CXXFLAGS += $(RELEASE_FLAGS) $(UNSAFE_RELEASE_FLAGS)
unsafe: LDFLAGS += -flto
unsafe: clean $(TARGET)

# Profile-guided optimization (Step 1: Generate profile)
profile-gen: CXXFLAGS += -O3 -march=native -fprofile-generate
profile-gen: LDFLAGS += -fprofile-generate
profile-gen: clean $(TARGET)
	@echo "Run ./$(TARGET) with typical workload, then run 'make profile-use'"

# Profile-guided optimization (Step 2: Use profile)
profile-use: CXXFLAGS += -O3 -march=native -fprofile-use -fprofile-correction
profile-use: LDFLAGS += -fprofile-use
profile-use: $(TARGET)
	@echo "Optimized build with PGO complete"

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR)
	rm -f *.profdata *.profraw  # Clean PGO files
	@echo "Clean complete"

# Phony targets
.PHONY: all debug release profile-gen profile-use run clean unsafe
