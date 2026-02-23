#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define _USE_MATH_DEFINES

#include <omp.h>

#include "cnpy.h"

#ifndef PROCESSING_H
#define PROCESSING_H

/**
*  File Header
*   Author: Nels Buhrley
*   Date: 2026-17-02
*   Description:
*/

class Material {
    int n; // Size of the lattice (n x n x n)
    float temperature;     // Temperature of the system
    float h;   // Total magnetization of the system
    int numIterations; // Number of iterations to run the simulation
    bool periodicBoundary;  // Whether to use periodic boundary conditions

    std::vector<std::vector<std::vector<uint8_t>>> spins;  // 3D array: [x][y][z]

    float exp_table[2][7]; // Precomputed exp(-ΔE/T) values

    public:
    Material(int n, float temperature, float magnetization, int numIterations, bool periodicBoundary)
        : n(n), temperature(temperature), h(magnetization), numIterations(numIterations), periodicBoundary(periodicBoundary) {
        // Initialize spins randomly to +1 or -1
        spins.resize(n, std::vector<std::vector<uint8_t>>(n, std::vector<uint8_t>(n)));
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                for (int z = 0; z < n; z++) {
                    spins[x][y][z] = (rand() % 2) * 2 - 1;  // Randomly +1 or -1
                }
            }
        }
    }

    Material(int n, float temperature, float magnetization, int numIterations, bool periodicBoundary, uint8_t initialSpinValue)
        : n(n), temperature(temperature), h(magnetization), numIterations(numIterations), periodicBoundary(periodicBoundary) {
        // Initialize all spins to the specified value (+1 or -1)
        if (initialSpinValue != 1 && initialSpinValue != -1) {
            throw std::invalid_argument("Initial spin value must be +1 or -1");
        }
        spins.resize(n, std::vector<std::vector<uint8_t>>(n, std::vector<uint8_t>(n, initialSpinValue)));

        int neighborValues[7] = { -6, -4, -2, 0, 2, 4, 6 };
        int spinValues[2] = { -1, 1 };

        // Precompute exp(-ΔE/T) for all possible ΔE values and spin states
        for (int s = 0; s < 2; s++) {
            for (int i = 0; i < 7; i++) {
                float deltaE = 2 * spinValues[s] * (neighborValues[i] + h);
                exp_table[s][i] = exp(-deltaE / temperature);
            }
        }
    }

    void iteration() {
        // Perform one iteration of algorithm
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                for (int z = (x + y) % 2; z < n; z += 2) {
                    // Calculate energy change if we flip this spin
                    int deltaE = 2 * spins[x][y][z] * (
                        spins(x + 1, y, z) + spins(x - 1, y, z) +
                        spins(x, y + 1, z) + spins(x, y - 1, z) +
                        spins(x, y, z + 1) + spins(x, y, z - 1)
                    );

                    // Decide whether to flip the spin
                    if (deltaE <= 0 || (rand() / static_cast<float>(RAND_MAX)) < exp(-deltaE / temperature)) {
                        spins[x][y][z] *= -1;  // Flip the spin
                    }
                }
            }
        }
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                for (int z = (x + y + 1) % 2 ; z < n; z += 2) {
                    // Calculate energy change if we flip this spin
                    int deltaE = 2 * spins[x][y][z] * (
                        spins(x + 1, y, z) + spins(x - 1, y, z) +
                        spins(x, y + 1, z) + spins(x, y - 1, z) +
                        spins(x, y, z + 1) + spins(x, y, z - 1)
                    );

                    // Decide whether to flip the spin
                    if (deltaE <= 0 || (rand() / static_cast<float>(RAND_MAX)) < exp(-deltaE / temperature)) {
                        spins[x][y][z] *= -1;  // Flip the spin
                    }
                }
            }
        }
    }

    void runSimulation() {
        for (int i = 0; i < numIterations; i++) {
            int thread_id = omp_get_thread_num();

            std::
            iteration();
        }
    }
}