#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <filesystem>

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

    std::vector<int8_t> spins;  // 3D array: [x][y][z]

    float deltaE_table[2][7]; // Precomputed energy changes for spin flips
    float exp_table[2][7]; // Precomputed exp(-ΔE/T) values

    std::uniform_real_distribution<float> distribution; // For random number generation
    std::mt19937 gen; // Mersenne Twister RNG
    uint32_t seed; // Seed for RNG

    public:
    Material(int n, float temperature, float magnetization, int numIterations, uint32_t seed)
        : n(n+2), temperature(temperature), h(magnetization), numIterations(numIterations), seed(seed) {
        establishRNG();
        initializeSpinsRandomly();
        precalculateEnergyTables();

    }

    Material(int n, float temperature, float magnetization, int numIterations, int8_t initialSpinValue, uint32_t seed)
        : n(n+2), temperature(temperature), h(magnetization), numIterations(numIterations), seed(seed) {
        establishRNG();
        initializeSpinsUniformly(initialSpinValue);
        precalculateEnergyTables();

    }

    inline int8_t getSpin(int x, int y, int z) {
        return spins[x * n * n + y * n + z];
    }

    inline void setSpin(int x, int y, int z, int8_t value) {
        spins[x * n * n + y * n + z] = value;
    }

    void initializeSpinsRandomly() {
        spins.resize(n * n * n);
        for (int x = 1; x < n-1; x++) {
            for (int y = 1; y < n-1; y++) {
                for (int z = 1; z < n-1; z++) {
                    setSpin(x, y, z, (distribution(gen) < 0.5 ? -1 : 1));  // Randomly +1 or -1
                }
            }
        }
    }

    void initializeSpinsUniformly(int8_t spinValue) {
        if (spinValue != 1 && spinValue != -1) {
            throw std::invalid_argument("Spin value must be +1 or -1");
        }
        spins.resize(n * n * n, spinValue);
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
        gen = std::mt19937(seed);
        distribution = std::uniform_real_distribution<float>(0.0, 1.0);


    }

    void flipSpin(int x, int y, int z) {
        // Calculate energy change if we flip this spin
        uint8_t neighborstate = (getSpin(x + 1, y, z) + getSpin(x - 1, y, z) +
                            getSpin(x, y + 1, z) + getSpin(x, y - 1, z) +
                            getSpin(x, y, z + 1) + getSpin(x, y, z - 1)) / 2 + 3; // Map neighbor sum from [-6,6] to [0,6]
        uint8_t spinState = (getSpin(x, y, z) + 1) / 2; // Map -1 to 0 and +1 to 1
        // Decide whether to flip the spin
        if (deltaE_table[spinState][neighborstate] <= 0 || (distribution(gen) < exp_table[spinState][neighborstate])) {
            setSpin(x, y, z, -getSpin(x, y, z));  // Flip the spin
        }
    }

    void iteration() {
        int x , y , z;
        for (x = 0; x < n; x++) {
            for (y = 0; y < n; y++) {
                setSpin(x, y, 0, getSpin(x, y, n-2));
                setSpin(x, y, n-1, getSpin(x, y, 1));
            }
        }
        for (x = 0; x < n; x++) {
            for (z = 0; z < n; z++) {
                setSpin(x, 0, z, getSpin(x, n-2, z));
                setSpin(x, n-1, z, getSpin(x, 1, z));
            }
        }
        for (y = 0; y < n; y++) {
            for (z = 0; z < n; z++) {
                setSpin(0, y, z, getSpin(n-2, y, z));
                setSpin(n-1, y, z, getSpin(1, y, z));
            }
        }
        // Perform one iteration of algorithm
        for (x = 1; x < n-1; x++) {
            for (y = 1; y < n-1; y++) {
                for (z = (x + y) % 2 + 1; z < n-1; z += 2) {
                    flipSpin(x, y, z);
                }
            }
        }
        for (x = 1; x < n-1; x++) {
            for (y = 1; y < n-1; y++) {
                for (z = (x + y + 1) % 2 + 1; z < n-1; z += 2) {
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

    double getAverageMagnetization() {
        double totalMagnetization = 0.0;
        for (int x = 1; x < n-1; x++) {
            for (int y = 1; y < n-1; y++) {
                for (int z = 1; z < n-1; z++) {
                    totalMagnetization += getSpin(x, y, z);
                }
            }
        }
        return totalMagnetization / ((n-2) * (n-2) * (n-2)); // Average magnetization per spin
    }
};

void saveResultsToNPZ(const std::vector<std::vector<double>>& avg_magnetizations, const std::vector<float>& temps, const std::vector<float>& h_values, const std::string& filename)
{
    // Convert 2D vector to 1D array for cnpy
    std::vector<double> magnetization_flat;
    for (const auto& row : avg_magnetizations) {
        magnetization_flat.insert(magnetization_flat.end(), row.begin(), row.end());
    }

    // Save the data to an NPZ file
    cnpy::npz_save(filename, "avg_magnetizations", magnetization_flat.data(), {h_values.size(), temps.size()}, "w");
    cnpy::npz_save(filename, "temperatures", temps.data(), {temps.size()}, "a");
    cnpy::npz_save(filename, "magnetic_fields", h_values.data(), {h_values.size()}, "a");
}


void runIsingSimulation(int n, int iterations, float hMin, float hMax, int numHSteps, float tempMin, float tempMax, int numTempSteps) {

    // 1. Generate the temperature range

    std::vector<float> temps(numTempSteps);

    float tempStep = (tempMax - tempMin) / (numTempSteps - 1);
    for (int i = 0; i < numTempSteps; i++) {
        temps[i] = tempMin + i * tempStep;
    }

    // 2. Generate the magnetic field range

    std::vector<float> h_values(numHSteps);

    float hStep = (hMax - hMin) / (numHSteps - 1);
    for (int i = 0; i < numHSteps; i++) {
        h_values[i] = hMin + i * hStep;
    }

    // 2. The Master RNG (Runs on a single thread)
    std::random_device rd;
    std::mt19937 master_gen(rd());
    std::uniform_int_distribution<uint32_t> seed_dist;

    // 3. Generate a unique, perfectly random integer seed for every temperature
    std::vector<uint32_t> thread_seeds(temps.size());
    for (long unsigned int i = 0; i < temps.size(); i++) {
        thread_seeds[i] = seed_dist(master_gen);
    }

    std::vector<std::vector<double>> avg_magnetizations(numHSteps, std::vector<double>(numTempSteps));

    // 4. Launch the parallel sweep
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (long unsigned int i = 0; i < temps.size(); i++) {
        for (long unsigned int j = 0; j < h_values.size(); j++) {
            Material material(n, temps[i], h_values[j], iterations, thread_seeds[i]);
            material.runSimulation();
            avg_magnetizations[j][i] = material.getAverageMagnetization();
            if (j == 0) {
                std::cout << temps[i] << " " << h_values[j] << std::endl;
            }
        }
    }

    // 5. Save results to NPZ file
    std::string filename = "output/ising_results.npz";
    saveResultsToNPZ(avg_magnetizations, temps, h_values, filename);

}




#endif // PROCESSING_H