#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include "processing.h"

/**
 * File Header
 *
 * Author: Nels Buhrley
 * Date: 2024-06-01
 * Description: Entry point for the 3D random-walk diffusion simulation.
 *   Creates a Space of particles starting at the origin, propagates them
 *   through a series of fixed-radius random steps confined to a cubic boundary,
 *   computes the RMS displacement at each time step, and saves both the
 *   per-step positions and the RMS data to CSV and NPZ output files.
 */


int main() {
    // Step size for each random walk move (meters).
    // Smaller values produce slower, more realistic diffusion.
    float movementRadius = 0.025f;  // 2.5 cm

    // Side length of the confining cube (meters).
    // Particles that would leave this boundary are reflected.
    float cubeSize = 1.f;  // 1 m cube

    // Number of time steps to simulate.
    // Increase for longer diffusion trajectories.
    size_t steps = 5;

    // Number of independent particles to simulate.
    // More particles give better statistical accuracy for RMS.
    size_t numPoints = 2500;

    // Base path for output files (relative to the project directory).
    // The simulation will append _Positions.csv, _RMS.csv, and .npz suffixes.
    std::string outputFilename = "output/defusion_output";

    // Construct the simulation space: all particles start at the origin (0, 0, 0).
    Space space(movementRadius, cubeSize, steps, numPoints);

    // Run the random-walk simulation across all time steps and particles.
    space.Propagate();

    // Compute root-mean-square displacement from the origin at each time step.
    space.calculateRMS();

    // Write position and RMS data to CSV files for Python plotting.
    space.saveToCSV(outputFilename);

    // Write the same data to a compressed NPZ file for efficient storage / NumPy loading.
    space.saveToNPZ(outputFilename);

    return 0;
}
