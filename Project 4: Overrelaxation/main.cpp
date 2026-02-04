#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include "Processing.h"


int main() {
    // Run app;  // Create the application controller
    // app.runSimulation();  // Launch the interactive simulation interface
    std::cout << "Starting overrelaxation simulation...\n";
    int N = 150;                     // Grid size
    double physicalDimensions = 1; // 1 meter cube
    PotentialField field(N, physicalDimensions);
    field.test_param();
    field.runSimulation(4000, 4e-10);  // Max 100000 iterations, tolerance 1e-20
    std::cout << "Simulation complete.\n";
    field.outputResultsToCSV("output.csv", "# Simulation results");
    field.exportTodotnpz("output.npz");
    std::cout << "Results saved to output.csv and output.npz\n";

    return 0;
};

// Using N = 400 on a intel i5 processor (4 cores 8 threads)
// CPU time ~ 2323 seconds or ~ 38.7 minutes
// 758% CPU usage observed (8 threads fully utilized most of the time)
// 5:09.44 wall time observed

/**
 * Decussions
 *
 * Solving Laplace's equation in 3D with SOR is computationally intensive.
 * The time complexity is O(N^3) per iteration, and convergence can take many iterations.
 * The overrelaxation parameter omega significantly affects convergence speed.
 * Optimal omega for SOR in 3D is approximately 2 / (1 + sqrt(2) + sin(pi / N)).
 *
 * Parallelization with OpenMP helps utilize multiple CPU cores, reducing wall time.
 *
 * standard SOR cannot be parallelized easily due to data dependencies.
 * However, red-black SOR or other techniques can be used to enable parallel updates.
 *
 * The calculation of the residual and the output of results is also parallelized to improve performance.
 *
 * Overall, using SOR with an optimal omega and parallelization makes solving large 3D potential problems feasible.
 *
 *
 * Final point:
 * Given a baseline of 17 seconds to solve the equation 
 */
