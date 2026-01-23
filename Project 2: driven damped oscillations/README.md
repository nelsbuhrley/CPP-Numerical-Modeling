# Project 2: Driven Damped Oscillations

A comprehensive C++ simulation of driven damped oscillator systems using numerical integration methods. The project features multiple integration schemes (RK4 and Euler-Cromer), automated plotting, and phase space analysis including Poincaré sections.

## Overview

This project simulates a driven damped pendulum/oscillator governed by the equation:
```
θ'' + (q/m)θ' + (g/L)sin(θ) = (F_D/m)cos(ω_D·t)
```

Where:
- `θ` = angle (radians)
- `q` = damping coefficient
- `m` = mass
- `g` = gravitational acceleration
- `L` = length
- `F_D` = driving force amplitude
- `ω_D` = driving frequency

The simulation can exhibit various behaviors including periodic motion, quasi-periodic motion, and chaos depending on the parameters.

## Project Structure

```
Project 2: driven damped oscillations/
├── main.cpp                  # Entry point - runs logic controller
├── Makefile                  # Build automation with multiple targets
├── ploting.py               # Basic plotting script
├── advanced_plotting.py     # 4-panel analysis tool (NEW)
├── README.md                # This file
├── bin/                     # Compiled executables
│   └── main
├── include/                 # Header files
│   ├── oscillator.h        # Oscillator class declarations
│   └── processing.h        # Integration methods & logic controller
├── src/                     # Implementation files
│   ├── oscillator.cpp      # Oscillator physics implementation
│   └── processing.cpp      # RK4, Euler-Cromer, and I/O
├── obj/                     # Object files (generated during build)
└── Output/                  # Simulation results and plots
    └── oscillator_output_*.csv
```

Due to the large size of output files they are not included in the repository

## Features

### Numerical Integration Methods
- **RK4 (Runge-Kutta 4th Order)**: High-accuracy symplectic integrator
- **Euler-Cromer**: Semi-implicit method optimized for oscillatory systems

### Analysis & Visualization
- **Time Series**: Angle and angular velocity vs. time
- **Phase Space**: Angular velocity vs. angle trajectories
- **Poincaré Sections**: Stroboscopic sampling at driver period
- **Angle Wrapping**: Automatic normalization to [-π, π]

### Simulation Modes
The `logic` class provides multiple simulation modes:
- Validation tests with known parameters
- Custom parameter simulations
- Automated data export to CSV
- Optional Python plotting integration

## Building the Project

### Prerequisites
- C++17 compatible compiler (g++, clang++)
- Python 3 (optional, for plotting)
- Required Python packages: `pandas`, `matplotlib`, `numpy`

### Build Commands

```bash
# Build the program (default target)
make

# Build and run
make run

# Debug build (no optimization, full symbols)
make debug

# Release build (maximum optimization)
make release

# Clean build artifacts
make clean

# Remove all generated files and directories
make distclean
```

### Manual Build (Alternative)
```bash
g++ -std=c++17 -Wall -Wextra -O2 -g -I./include \
    main.cpp src/oscillator.cpp src/processing.cpp \
    -o bin/main
```

## Running the Simulation

### Basic Execution
```bash
./bin/main
```

The program will prompt you to select a simulation mode and enter parameters.

### Output
Simulation results are saved to `Output/oscillator_output_N.csv` with columns:
- `Time` (seconds)
- `Angle` (radians)
- `AngularVelocity` (radians/second)

## Visualization

### Basic Plotting
```bash
python3 ploting.py
```

### Advanced 4-Panel Analysis
```bash
# Command-line mode
python3 advanced_plotting.py <csv_file> <max_time> <driver_frequency>

# Example
python3 advanced_plotting.py "Output/oscillator_output_0.csv" 180 0.666

# Interactive mode (prompts for parameters)
python3 advanced_plotting.py
```

The advanced plotting tool generates:
1. **Angle vs Time (0-180s)** - Full unwrapped angle evolution
2. **Wrapped Angle vs Time (0-180s)** - Angle normalized to [-π, π]
3. **Phase Space (0-180s)** - Angular velocity vs. wrapped angle
4. **Poincaré Section (full dataset)** - Stroboscopic sampling at driver period

Output: `Output/oscillator_analysis_N.png` (300 DPI, matching input file number)

## Code Structure

### Main Components

#### `oscillator.h/cpp`
Defines the oscillator physics model and state management.

#### `processing.h/cpp`
- **`integrator` namespace**: Contains `rk4()` and `euler_chromer()` integration functions
- **`logic` class**: High-level simulation controller
  - `run()`: Interactive simulation launcher
  - `runValidationTest()`: Predefined test cases
  - `runCustomSimulation()`: User-defined parameters
  - `outputResults()`: CSV export with metadata

#### `main.cpp`
Minimal entry point that instantiates and runs the `logic` controller.

## Customization

### Modify Oscillator Parameters
Edit the oscillator constructor call in `processing.cpp`:
```cpp
testOscillator osc(mass, length, dampingCoeff, initialAngle,
                   initialVelocity, drivingForce, drivingFreq);
```

### Change Integration Method
In `processing.cpp`, switch between:
```cpp
auto path = integrator::rk4(state, derivFunc, stopCondition, timeStep);
// or
auto path = integrator::euler_chromer(state, derivFunc, stopCondition, timeStep);
```

### Adjust Time Step & Duration
Modify simulation parameters:
```cpp
double timeStep = 0.0001;  // Smaller = more accurate, slower
double maxTime = 60.0;     // Simulation duration (seconds)
```

## Example Interesting Parameter Sets

### Chaotic Regime (Period Doubling)
- Damping: q/m = 0.5
- Natural frequency: (g/L)^0.5 = 9.8^0.5 ≈ 3.13 rad/s
- Driving force: F_D/m = 1.2
- Driving frequency: ω_D = 2/3 rad/s

### Periodic Motion
- Low driving force (F_D/m < 0.5)
- Driving frequency near resonance

### Resonance
- ω_D ≈ √(g/L) - observe amplitude growth

## Tips for Exploration

1. **Stability**: Start with small time steps (1e-4) and verify energy conservation
2. **Chaos Detection**: Look for sensitivity to initial conditions in Poincaré sections
3. **Parameter Sweeps**: Vary driving frequency or amplitude systematically
4. **Long Simulations**: Chaotic behavior may take hundreds of periods to emerge

## Troubleshooting

### Compilation Errors
- Ensure C++17 support: `-std=c++17`
- Check include paths: `-I./include`
- Verify all source files are listed in Makefile

### Numerical Issues
- Reduce time step if trajectories diverge
- Check initial conditions are physical
- Verify stop condition isn't too restrictive

### Plotting Issues
- Ensure Python environment has required packages
- Check CSV file path is correct
- Verify driver frequency matches simulation parameters

## Further Development

Potential enhancements:
- GPU acceleration for parameter sweeps
- Lyapunov exponent calculation
- Bifurcation diagram generation
- Multiple pendulum coupling
- Energy analysis and dissipation tracking

## References

- Runge-Kutta Methods: Numerical Recipes in C++
- Driven Pendulum Dynamics: Strogatz, "Nonlinear Dynamics and Chaos"
- Poincaré Sections: Alligood, Sauer, Yorke, "Chaos: An Introduction"

---

**License**: MIT (or specify your own)
**Author**: [Your Name]
**Last Updated**: January 2026
