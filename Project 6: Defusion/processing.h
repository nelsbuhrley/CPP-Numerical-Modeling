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
 *File Header
 *
 *Author: Nels Buhrley
 *Date: 2026-18-02
 *Description:

 *
 */

class Space {
   public:
    std::vector<std::vector<std::array<float, 3>>> points;  // 3D points
    std::vector<float> rRMS;  // Root mean square distance from origin at each step (for analysis)
    float movementRadius;     // Radius of movement for each point
    float cubeHalfSize;       // Half the size of the cube for boundary check
    size_t steps;             // Number of time steps
    size_t numPoints;         // Number of points in the space

    Space(float movementRadius, float cubeSize, size_t steps, size_t numPoints)
        : points(static_cast<size_t>(steps),
                 std::vector<std::array<float, 3>>(numPoints, std::array<float, 3>{0.f, 0.f, 0.f})),
          rRMS(steps, 0.f),
          movementRadius(movementRadius),
          cubeHalfSize(cubeSize / 2.f),
          steps(steps),
          numPoints(numPoints) {}

    void movePoint(size_t step, size_t index, std::default_random_engine& generator,
                   std::uniform_real_distribution<float>& unitDist) {
        const float u = unitDist(generator);  // Random number for angle
        const float v = unitDist(generator);  // Random number for angle

        const float theta = 2.f * static_cast<float>(M_PI) * u;  // Azimuthal angle
        const float cosPhi =
            1.f - 2.f * v;  // Cosine of polar angle (uniform distribution on sphere)
        const float sinPhi = std::sqrt(1.f - cosPhi * cosPhi);
        const float r = movementRadius * std::cbrt(unitDist(generator));
        // const float r = movementRadius * unitDist(generator);
        // const float r = movementRadius;  // Fixed radius
        const float rSinPhi = r * sinPhi;

        std::array<float, 3>& pl = points[step - 1][index];
        std::array<float, 3>& p = points[step][index];

        p[0] = pl[0] + rSinPhi * std::cos(theta);
        p[1] = pl[1] + rSinPhi * std::sin(theta);
        p[2] = pl[2] + r * cosPhi;
        for (int i = 0; i < 3; ++i) {
            if (p[i] < -cubeHalfSize)
                p[i] = -2.f * cubeHalfSize - p[i];
            else if (p[i] > cubeHalfSize)
                p[i] = 2.f * cubeHalfSize - p[i];
        }
    }

    void Propagate() {
#pragma omp parallel
        {
            std::default_random_engine generator(
                omp_get_thread_num() +
                static_cast<unsigned int>(
                    std::chrono::system_clock::now().time_since_epoch().count()));
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

    void calculateRMS() {
#pragma omp parallel for schedule(static)
        for (size_t step = 0; step < steps; ++step) {
            float sumSq = 0.f;
            for (size_t p = 0; p < numPoints; ++p) {
                const auto& point = points[step][p];
                sumSq += point[0] * point[0] + point[1] * point[1] + point[2] * point[2];
            }
            rRMS[step] = std::sqrt(sumSq / numPoints);
        }
    }

    void savePosisionsToCSV(const std::string filename) {
        // Build metadata header
        std::string header;
        header += "# Cube Size: " + std::to_string(cubeHalfSize * 2.f) + "m,\n";
        header += "# Movement Radius: " + std::to_string(movementRadius) + "m\n";
        header += "# Steps: " + std::to_string(steps) + "\n";
        header += "# Number of Points: " + std::to_string(numPoints) + "\n";

        for (size_t p = 0; p < numPoints; ++p) {
            header += "," + std::to_string(p + 1) + "x" + "," + std::to_string(p + 1) + "y" + "," +
                      std::to_string(p + 1) + "z";
        }
        header += "\n";

        // Build each data row in parallel into a pre-allocated cache
        std::vector<std::string> rows(steps);
#pragma omp parallel for schedule(static)
        for (size_t step = 0; step < steps; ++step) {
            std::string row = std::to_string(step + 1);
            for (size_t p = 0; p < numPoints; ++p) {
                const auto& point = points[step][p];
                row += "," + std::to_string(point[0]) + "," + std::to_string(point[1]) + "," +
                       std::to_string(point[2]);
            }
            row += "\n";
            rows[step] = std::move(row);
        }

        // Single sequential write: header then all cached rows
        std::ofstream file(filename);
        file << header;
        for (const auto& row : rows) {
            file << row;
        }


    }

    void saveRMSToCSV(const std::string filename) {
        std::ofstream file(filename);
        file << "Step,RMS\n";
        for (size_t step = 0; step < steps; ++step) {
            file << (step + 1) << "," << rRMS[step] << "\n";
        }
    }

    void saveToCSV(const std::string filename) {
        savePosisionsToCSV(filename + "_Positions.csv");
        saveRMSToCSV(filename + "_RMS.csv");
    }

    void saveToNPZ(const std::string filename) {
        std::string npzFilename = filename + ".npz";
        // Flatten the 3D points into a 1D array for cnpy
        std::vector<float> flatPoints(steps * numPoints * 3);
#pragma omp parallel for collapse(2) schedule(static)
        for (size_t step = 0; step < steps; ++step) {
            for (size_t p = 0; p < numPoints; ++p) {
                const auto& point = points[step][p];
                size_t idx = (step * numPoints + p) * 3;
                flatPoints[idx] = point[0];
                flatPoints[idx + 1] = point[1];
                flatPoints[idx + 2] = point[2];
            }
        }

        std::vector<float> metadata = {
            cubeHalfSize * 2.f,
            movementRadius,
        };

        cnpy::npz_save(npzFilename, "points", flatPoints.data(), {steps, numPoints, 3}, "w");
        cnpy::npz_save(npzFilename, "rRMS", rRMS.data(), {steps}, "a");
        cnpy::npz_save(npzFilename, "metadata", metadata, "a");
    }

};

#endif
