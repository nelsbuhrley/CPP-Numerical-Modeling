# CPP_Workspace Overview

## Purpose
A collection of numerical physics simulations and mathematical explorations written in C++17. Projects range from classical mechanics and celestial dynamics to number theory, all with Python-based visualization.

## Quick Start
1. Navigate into a project directory.
2. Run `make` to build
3. Run the executable from `bin/`.
4. Run the accompanying `plotting.py` to visualize results.

## Build & Toolchain
- **Compiler**: Apple Clang++ (C++17), g++, gcc, or icpx depending on requirements and system (Fulton supercomputer or personal machine)
- **Build System**: `make` (per-project Makefiles)
- **Parallelization**: OpenMP (used in Projects 4–6)
- **Output formats**: CSV and NPZ (via `cnpy`)
- **Visualization**: Python / Matplotlib

## Projects

### Project 1: Realistic Projectile Motion
Physics simulation of projectile trajectories including gravity, air drag, wind, and spin-induced Magnus force. Uses custom `Vector3D`/`Vector4D` classes and Runge-Kutta integration.

### Project 2: Driven Damped Oscillations
Simulates a driven damped pendulum governed by a nonlinear ODE. Supports RK4 and Euler-Cromer integrators, phase-space analysis, and Poincaré sections. Can exhibit periodic, quasi-periodic, or chaotic motion depending on parameters.

### Project 3: Celestial Dynamics
Interactive N-body gravitational simulation. Prompts the user to configure bodies and integration parameters, then outputs trajectories for Python visualization.

### Project 4: Overrelaxation
Solves Laplace's equation on a 3D grid using Red Black Successive Over-Relaxation (SOR). Parallelized with OpenMP. Outputs the resulting potential field to CSV and NPZ.

### Project 5: Oscillations on a String
Finite-difference simulation of transverse string oscillations. Supports Gaussian, sine, and natural-mode initial conditions. FFT analysis (via KissFFT) produces frequency spectra. Parallelized with OpenMP.

### Project 6: Diffusion
3D random-walk diffusion simulation tracking RMS displacement of particles over time in a bounded cube. Outputs per-step positions and RMS data to CSV and NPZ.

## Personal Projects

### Personal Project 1: 3n+1 (Collatz Conjecture)
Explores the Collatz sequence up to 1,000,000, identifies the number requiring the most steps, generates CSV data, and plots results with Python.

### Personal Project 2: Idelic Numbers
Multithreaded sieve-based search for Euler's idoneal numbers up to a configurable limit, using `std::thread` for parallelism.

## Reference
- `reference/CPP_BASICS.md` — C++ syntax reference
- `reference/FUNCTIONS_REFERENCE.md` — Common function patterns
- `reference/NUMERICAL_PHYSICS.md` — Numerical methods notes
- `reference/OPTIMIZATION_TIPS.md` — Performance guidelines
- `include/cnpy.h` — NPZ file I/O library
- `include/vector3d.h` — Shared 3D vector utilities

## Shortcuts
- **F5**: Build and Run (active project)
- **Cmd+Shift+B**: Build Only
- **Ctrl+\`**: Toggle Terminal
