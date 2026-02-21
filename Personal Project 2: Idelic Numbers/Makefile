# Makefile for Celestial Dynamics Project
# Header-only implementation: all code is in include/*.h files
#
# Usage:
#   make              # Build the program
#   make clean        # Remove build artifacts
#   make run          # Build and run
#   make debug        # Build with debug symbols
#   make release      # Build optimized version

# Compiler and flags
CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude
DEBUG_FLAGS = -O0 -g
RELEASE_FLAGS = -O3 -DNDEBUG

# Directories
INCLUDE_DIR = include
BIN_DIR = bin

# Header-only project: only main.cpp needs compilation
# All other code is in header files (Objects.h, processing.h, run.h)
SOURCES = main.cpp
HEADERS = $(wildcard $(INCLUDE_DIR)/*.h)

# Executable name
TARGET = $(BIN_DIR)/main

# Default target
all: $(TARGET)

# Build the program (recompile if any header changes)
$(TARGET): $(SOURCES) $(HEADERS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES)
	@echo "Build complete: $(TARGET)"

# Debug build
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean $(TARGET)

# Release build
release: CXXFLAGS += $(RELEASE_FLAGS)
release: clean $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BIN_DIR)
	@echo "Clean complete"

# Phony targets
.PHONY: all debug release run clean
