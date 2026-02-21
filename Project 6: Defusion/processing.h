#define _USE_MATH_DEFINES
#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cnpy.h"

#ifndef PROCESSING_H
#define PROCESSING_H

/**
 * File Header
 *
 * Author: Nels Buhrley
 * Date: 2026-18-02
 * Description: Defines the Space class, which simulates 3D random-walk diffusion
 *   of N particles confined inside a cube. Each particle takes fixed-radius steps
 *   in a uniformly random direction on the unit sphere. Boundary collisions are
 *   handled by reflective (mirror) conditions. The class is parallelized with
 *   OpenMP and supports CSV and NPZ output.
 *
 * 3D random walk diffusion is a fundamental physical process with applications in
 * fields like physics, chemistry, and biology. This simulation models particles
 * undergoing Brownian motion within a confined cubic volume.
 *
 * Despite the simplicity of the underlying algorithm, monte carlo simulations of diffusion
 * can be suprisingly accurate and useful for understanding real-world phenomena involving
 * particles.
/**
 * @class Space
 * @brief Simulates 3D random-walk diffusion of particles confined in a cubic volume.
 *
 * All particles start at the origin. At each time step every particle moves by
 * exactly `movementRadius` in a uniformly random direction. If a particle would
 * cross a wall of the cube it is reflected back (mirror boundary condition).
 * After simulation, the RMS displacement from the origin is computed per step.
 */
class Space {
   public:
    // positions[step][particle] = {x, y, z}
    // Outer dimension is time step; inner dimension is particle index.
    std::vector<std::vector<std::array<float, 3>>> points;

    // RMS displacement from the origin at each time step, computed by calculateRMS().
    std::vector<float> rRMS;

    float movementRadius;  // Fixed step length for each random walk move (meters).
    float cubeHalfSize;    // Half the cube side length; used for boundary checks (meters).
    size_t steps;          // Total number of time steps to simulate (including step 0 at origin).
    size_t numPoints;      // Number of independent particles in the simulation.

    /**
     * @brief Constructs the simulation space with all particles at the origin.
     * @param movementRadius  Fixed step length per move (meters).
     * @param cubeSize        Full side length of the confining cube (meters).
     * @param steps           Number of time steps (positions stored for each).
     * @param numPoints       Number of particles to simulate.
     */
    Space(float movementRadius, float cubeSize, size_t steps, size_t numPoints)
        : points(static_cast<size_t>(steps), std::vector<std::array<float, 3>>(numPoints, std::array<float, 3>{0.f, 0.f, 0.f})),
          rRMS(steps, 0.f),
          movementRadius(movementRadius),
          cubeHalfSize(cubeSize / 2.f),
          steps(steps),
          numPoints(numPoints) {}

    /**
     * @brief Advances a single particle by one random step and applies reflective boundaries.
     *
     * A direction is sampled uniformly on the unit sphere using the standard
     * (u, v) -> (theta, phi) mapping, ensuring no polar bias:
     *   theta  = 2*pi*u          (azimuthal angle, uniform in [0, 2*pi))
     *   cosPhi = 1 - 2*v         (cosine of polar angle, uniform in [-1, 1])
     * The new position is the previous position plus the displacement vector.
     * If any coordinate leaves [-cubeHalfSize, cubeHalfSize] it is reflected.
     *
     * @param step       Current time step index (must be >= 1).
     * @param index      Particle index within the step.
     * @param generator  Per-thread random engine.
     * @param unitDist   Uniform distribution over [0, 1).
     */
    void movePoint(size_t step, size_t index, std::default_random_engine& generator, std::uniform_real_distribution<float>& unitDist) {
        const float u = unitDist(generator);  // Uniform sample for azimuthal angle
        const float v = unitDist(generator);  // Uniform sample for polar angle cosine

        const float theta = 2.f * static_cast<float>(M_PI) * u;  // Azimuthal angle [0, 2*pi)
        const float cosPhi = 1.f - 2.f * v;                      // cos(polar angle), uniform in [-1, 1]
        const float sinPhi = std::sqrt(1.f - cosPhi * cosPhi);   // sin(polar angle) >= 0

        // Alternative step-length options (disabled):
        // const float r = movementRadius * std::cbrt(unitDist(generator));  // volume-uniform radius
        // const float r = movementRadius * unitDist(generator);              // linear-uniform radius
        const float r = movementRadius;    // Fixed-radius step (current mode)
        const float rSinPhi = r * sinPhi;  // Pre-computed for x/y components

        // References to the previous and current position of this particle
        std::array<float, 3>& pl = points[step - 1][index];  // Previous position
        std::array<float, 3>& p = points[step][index];       // Current (new) position

        // Compute the new position by adding the displacement vector
        p[0] = pl[0] + rSinPhi * std::cos(theta);  // x component
        p[1] = pl[1] + rSinPhi * std::sin(theta);  // y component
        p[2] = pl[2] + r * cosPhi;                 // z component

        // Reflective boundary condition: mirror any coordinate that exits the cube
        for (int i = 0; i < 3; ++i) {
            if (p[i] < -cubeHalfSize)
                p[i] = -2.f * cubeHalfSize - p[i];  // Reflect off the negative wall
            else if (p[i] > cubeHalfSize)
                p[i] = 2.f * cubeHalfSize - p[i];  // Reflect off the positive wall
        }
    }

    /**
     * @brief Runs the full random-walk simulation across all steps and particles.
     *
     * The outer (step) loop is sequential because step N depends on step N-1.
     * The inner (particle) loop is parallelized with OpenMP. Each thread gets
     * its own RNG seeded by thread ID + wall-clock time to avoid correlation.
     * An implicit barrier at the end of each `omp for` ensures all particles
     * finish step N before any thread begins step N+1.
     */
    void Propagate() {
#pragma omp parallel
        {
            // Seed each thread's RNG differently to avoid identical sequences
            std::default_random_engine generator(omp_get_thread_num() +
                                                 static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
            std::uniform_real_distribution<float> unitDist(0.f, 1.f);

            // Step loop must remain sequential: each step depends on step-1.
            // Parallelize only the inner points loop; the implicit barrier after
            // each omp-for ensures all threads finish step N before step N+1.
            for (size_t step = 1; step < steps; ++step) {
#pragma omp for schedule(static)
                for (size_t i = 0; i < numPoints; ++i) {
                    movePoint(step, i, generator, unitDist);
                }
            }
        }
    }

    /**
     * @brief Computes the RMS displacement from the origin at each time step.
     *
     * For step t:  rRMS[t] = sqrt( mean( x^2 + y^2 + z^2 ) over all particles )
     * Each step is independent so all steps are computed in parallel.
     */
    void calculateRMS() {
#pragma omp parallel for schedule(static)
        for (size_t step = 0; step < steps; ++step) {
            float sumSq = 0.f;
            for (size_t p = 0; p < numPoints; ++p) {
                const auto& point = points[step][p];
                // Accumulate squared distance from origin for this particle
                sumSq += point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
            }
            // RMS = sqrt of the mean squared displacement
            rRMS[step] = std::sqrt(sumSq / numPoints);
        }
    }

    /**
     * @brief Writes particle positions to a CSV file.
     *
     * Format: one row per time step. Columns are:
     *   step, 1x, 1y, 1z, 2x, 2y, 2z, ...
     * Rows are built in parallel and written sequentially to avoid I/O races.
     *
     * @param filename  Output file path (should end in .csv).
     */
    void savePosisionsToCSV(const std::string filename) {
        // Build the comment header with simulation parameters
        std::string header;
        header += "# Cube Size: " + std::to_string(cubeHalfSize * 2.f) + "m,\n";
        header += "# Movement Radius: " + std::to_string(movementRadius) + "m\n";
        header += "# Steps: " + std::to_string(steps) + "\n";
        header += "# Number of Points: " + std::to_string(numPoints) + "\n";

        // Column labels: step index followed by Nx, Ny, Nz for each particle N
        for (size_t p = 0; p < numPoints; ++p) {
            header += "," + std::to_string(p + 1) + "x" + "," + std::to_string(p + 1) + "y" + "," + std::to_string(p + 1) + "z";
        }
        header += "\n";

        // Build each data row in parallel into a pre-allocated vector of strings
        // to avoid serializing string formatting through a single thread.
        std::vector<std::string> rows(steps);
#pragma omp parallel for schedule(static)
        for (size_t step = 0; step < steps; ++step) {
            std::string row = std::to_string(step + 1);  // First column: 1-based step number
            for (size_t p = 0; p < numPoints; ++p) {
                const auto& point = points[step][p];
                row += "," + std::to_string(point[0]) + "," + std::to_string(point[1]) + "," + std::to_string(point[2]);
            }
            row += "\n";
            rows[step] = std::move(row);
        }

        // Single sequential write pass: header then all pre-built rows in order
        std::ofstream file(filename);
        file << header;
        for (const auto& row : rows) {
            file << row;
        }
    }

    /**
     * @brief Writes the per-step RMS displacement to a two-column CSV file.
     *
     * Format:  Step,RMS  (1-based step index)
     *
     * @param filename  Output file path (should end in .csv).
     */
    void saveRMSToCSV(const std::string filename) {
        std::ofstream file(filename);
        file << "Step,RMS\n";
        for (size_t step = 0; step < steps; ++step) {
            file << (step + 1) << "," << rRMS[step] << "\n";
        }
    }

    /**
     * @brief Convenience wrapper that saves both positions and RMS to CSV.
     *
     * Produces:  <filename>_Positions.csv  and  <filename>_RMS.csv
     *
     * @param filename  Base output path (no extension).
     */
    void saveToCSV(const std::string filename) {
        savePosisionsToCSV(filename + "_Positions.csv");
        saveRMSToCSV(filename + "_RMS.csv");
    }

    /**
     * @brief Saves simulation data to a compressed NumPy NPZ archive.
     *
     * Arrays stored in the NPZ file:
     *   "points"   : float32 array of shape (steps, numPoints, 3) — particle positions.
     *   "rRMS"     : float32 array of shape (steps,)              — RMS displacement per step.
     *   "metadata" : float32 array [cubeSize, movementRadius]     — simulation parameters.
     *
     * The 3D points array is first flattened row-major into a 1D buffer (parallelized),
     * then saved via cnpy which handles the zip/npy encoding.
     *
     * @param filename  Base output path (no extension); ".npz" is appended.
     */
    void saveToNPZ(const std::string filename) {
        std::string npzFilename = filename + ".npz";

        // Flatten points[step][particle][xyz] into a contiguous 1D buffer
        // required by cnpy::npz_save. Parallelized over steps and particles.
        std::vector<float> flatPoints(steps * numPoints * 3);
#pragma omp parallel for collapse(2) schedule(static)
        for (size_t step = 0; step < steps; ++step) {
            for (size_t p = 0; p < numPoints; ++p) {
                const auto& point = points[step][p];
                size_t idx = (step * numPoints + p) * 3;  // Row-major index into flat buffer
                flatPoints[idx] = point[0];               // x
                flatPoints[idx + 1] = point[1];           // y
                flatPoints[idx + 2] = point[2];           // z
            }
        }

        // Small metadata array so the NPZ is self-describing
        std::vector<float> metadata = {
            cubeHalfSize * 2.f,  // Full cube side length (meters)
            movementRadius,      // Step length (meters)
        };

        // Write to NPZ: "w" creates/overwrites the archive; "a" appends additional arrays
        cnpy::npz_save(npzFilename, "points", flatPoints.data(), {steps, numPoints, 3}, "w");
        cnpy::npz_save(npzFilename, "rRMS", rRMS.data(), {steps}, "a");
        cnpy::npz_save(npzFilename, "metadata", metadata, "a");
    }
};

#endif
