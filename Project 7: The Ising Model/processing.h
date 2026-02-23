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

    std::vector<std::vector<std::vector<int8_t>>> spins;  // 3D array: [x][y][z]

    float deltaE_table[2][7]; // Precomputed energy changes for spin flips
    float exp_table[2][7]; // Precomputed exp(-ΔE/T) values

    std::uniform_real_distribution<float> distribution; // For random number generation
    std::mt19937 gen; // Mersenne Twister RNG

    public:
    Material(int n, float temperature, float magnetization, int numIterations, bool periodicBoundary)
        : n(n), temperature(temperature), h(magnetization), numIterations(numIterations), periodicBoundary(periodicBoundary) {
        // Initialize spins randomly to +1 or -1
        spins.resize(n, std::vector<std::vector<int8_t>>(n, std::vector<int8_t>(n)));
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                for (int z = 0; z < n; z++) {
                    spins[x][y][z] = (rand() % 2) * 2 - 1;  // Randomly +1 or -1
                }
            }
        }
        precalculateEnergyTables();
    }

    Material(int n, float temperature, float magnetization, int numIterations, bool periodicBoundary, int8_t initialSpinValue)
        : n(n), temperature(temperature), h(magnetization), numIterations(numIterations), periodicBoundary(periodicBoundary) {
        // Initialize all spins to the specified value (+1 or -1)
        if (initialSpinValue != 1 && initialSpinValue != -1) {
            throw std::invalid_argument("Initial spin value must be +1 or -1");
        }
        spins.resize(n, std::vector<std::vector<int8_t>>(n, std::vector<int8_t>(n, initialSpinValue)));

        precalculateEnergyTables();
    }

    void precalculateEnergyTables() {
        int8_t neighborValues[7] = { -6, -4, -2, 0, 2, 4, 6 };
        int8_t spinValues[2] = { -1, 1 };

        // Precompute exp(-ΔE/T) for all possible ΔE values and spin states
        for (int s = 0; s < 2; s++) {
            for (int i = 0; i < 7; i++) {
                float deltaE = 2 * spinValues[s] * (neighborValues[i] + h);
                deltaE_table[s][i] = deltaE;
                exp_table[s][i] = exp(-deltaE / temperature);
            }
        }
    }

    void establishRNG() {
        // Thread-safe random number generator setup
        std::random_device rd;
        gen = std::mt19937(rd());
        distribution = std::uniform_real_distribution<float>(0.0, 1.0);


    }

    void flipSpin(int x, int y, int z) {
        // Calculate energy change if we flip this spin
        uint8_t neighborstate = (spins(x + 1, y, z) + spins(x - 1, y, z) +
                            spins(x, y + 1, z) + spins(x, y - 1, z) +
                            spins(x, y, z + 1) + spins(x, y, z - 1)) / 2 + 3; // Map neighbor sum from [-6,6] to [0,6]
        uint8_t spinState = (spins[x][y][z] + 1) / 2; // Map -1 to 0 and +1 to 1
        // Decide whether to flip the spin
        if (deltaE_table[spinState][neighborstate] <= 0 || (distribution(gen) < exp_table[spinState][neighborstate])) {
            spins[x][y][z] *= -1;  // Flip the spin
        }
    }

    void iteration() {
        // Perform one iteration of algorithm
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                for (int z = (x + y) % 2; z < n; z += 2) {
                    flipSpin(x, y, z);
                }
            }
        }
        for (int x = 0; x < n; x++) {
            for (int y = 0; y < n; y++) {
                for (int z = (x + y + 1) % 2 ; z < n; z += 2) {
                    flipSpin(x, y, z);
                }
            }
        }
    }

    void runSimulation() {
        for (int i = 0; i < numIterations; i++) {

            iteration();
        }
    }
}