#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
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
   public:
    int n;              // Size of the lattice (n x n x n)
    int actual_n;       // Actual size of the lattice excluding boundaries
    int N;              // Total number of spins in the lattice (actual_n^3)
    float temperature;  // Temperature of the system
    float h;            // Total external magnetic field of the system
    int numIterations;  // Number of iterations to run the simulation

    float averageMagnetization;         // Running average magnetization for current temperature and magnetic field
    float averageMagnetizationSquared;  // Running average of magnetization squared for current temperature and magnetic field
    float averageAbsMagnetization;      // Running average of absolute magnetization for current temperature and magnetic field

    float magneticSusceptibility;  // Magnetic susceptibility for current temperature and magnetic field

    int currentTotalMagnetization;  // Current total magnetization of the lattice (used for incremental updates)

    std::vector<int8_t> spins;  // 3D array: [x][y][z]

    float deltaE_table[2][7];  // Precomputed energy changes for spin flips
    float exp_table[2][7];     // Precomputed exp(-ΔE/T) values

    std::uniform_real_distribution<float> distribution;  // For random number generation
    std::mt19937 gen;                                    // Mersenne Twister RNG
    uint32_t seed;                                       // Seed for RNG

   public:
    Material(int n, float temperature, float magnetization, int numIterations, uint32_t seed)
        : n(n + 2), actual_n(n), N(actual_n * actual_n * actual_n), temperature(temperature), h(magnetization), numIterations(numIterations), seed(seed) {
        establishRNG();
        initializeSpinsRandomly();
        precalculateEnergyTables();
    }

    Material(int n, float temperature, float magnetization, int numIterations, int8_t initialSpinValue, uint32_t seed)
        : n(n + 2), actual_n(n), N(actual_n * actual_n * actual_n), temperature(temperature), h(magnetization), numIterations(numIterations), seed(seed) {
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
        currentTotalMagnetization = 0;
        spins.resize(n * n * n);
        for (int x = 1; x < n - 1; x++) {
            for (int y = 1; y < n - 1; y++) {
                for (int z = 1; z < n - 1; z++) {
                    setSpin(x, y, z, (distribution(gen) < 0.5 ? -1 : 1));  // Randomly +1 or -1
                    currentTotalMagnetization += getSpin(x, y, z);
                }
            }
        }
    }

    void initializeSpinsUniformly(int8_t spinValue) {
        if (spinValue != 1 && spinValue != -1) {
            throw std::invalid_argument("Spin value must be +1 or -1");
        }
        spins.resize(n * n * n, spinValue);
        currentTotalMagnetization = spinValue * N;
    }

    void precalculateEnergyTables() {
        int8_t neighborValues[7] = {-6, -4, -2, 0, 2, 4, 6};
        int8_t spinValues[2] = {-1, 1};

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
        uint8_t neighborstate = (getSpin(x + 1, y, z) + getSpin(x - 1, y, z) + getSpin(x, y + 1, z) + getSpin(x, y - 1, z) + getSpin(x, y, z + 1) +
                                 getSpin(x, y, z - 1)) /
                                    2 +
                                3;                       // Map neighbor sum from [-6,6] to [0,6]
        uint8_t spinState = (getSpin(x, y, z) + 1) / 2;  // Map -1 to 0 and +1 to 1
        // Decide whether to flip the spin
        if (deltaE_table[spinState][neighborstate] <= 0 || (distribution(gen) < exp_table[spinState][neighborstate])) {
            setSpin(x, y, z, -getSpin(x, y, z));  // Flip the spin
            currentTotalMagnetization += 2 * getSpin(x, y, z);  // Update total magnetization
        }
    }

    void iteration() {
        int x, y, z;
        for (x = 0; x < n; x++) {
            for (y = 0; y < n; y++) {
                setSpin(x, y, 0, getSpin(x, y, n - 2));
                setSpin(x, y, n - 1, getSpin(x, y, 1));
            }
        }
        for (x = 0; x < n; x++) {
            for (z = 0; z < n; z++) {
                setSpin(x, 0, z, getSpin(x, n - 2, z));
                setSpin(x, n - 1, z, getSpin(x, 1, z));
            }
        }
        for (y = 0; y < n; y++) {
            for (z = 0; z < n; z++) {
                setSpin(0, y, z, getSpin(n - 2, y, z));
                setSpin(n - 1, y, z, getSpin(1, y, z));
            }
        }
        // Perform one iteration of algorithm
        for (x = 1; x < n - 1; x++) {
            for (y = 1; y < n - 1; y++) {
                for (z = (x + y) % 2 + 1; z < n - 1; z += 2) {
                    flipSpin(x, y, z);
                }
            }
        }
        for (x = 1; x < n - 1; x++) {
            for (y = 1; y < n - 1; y++) {
                for (z = (x + y + 1) % 2 + 1; z < n - 1; z += 2) {
                    flipSpin(x, y, z);
                }
            }
        }
    }

    void runSimulation() {
        float sum_magnetization = 0.0;
        float sum_magnetization_squared = 0.0;
        float sum_abs_magnetization = 0.0;

        for (int i = 0; i < 100; i++) {
            iteration();
        }
        for (int i = 0; i < numIterations; i++) {
            iteration();
            float currentMagnetization = (float)currentTotalMagnetization / N;
            sum_magnetization += currentMagnetization;
            sum_magnetization_squared += currentMagnetization * currentMagnetization;
            sum_abs_magnetization += std::abs(currentMagnetization);
        }
        averageMagnetization = sum_magnetization / numIterations;
        averageAbsMagnetization = sum_abs_magnetization / numIterations;
        averageMagnetizationSquared = sum_magnetization_squared / numIterations;
    }

    void MagneticSusceptibility() {
        magneticSusceptibility = actual_n * actual_n * actual_n * (averageMagnetizationSquared - averageAbsMagnetization * averageAbsMagnetization) / temperature;
    }
};

class Simulation {
   public:
    std::vector<std::vector<double>> avg_magnetizations;
    std::vector<std::vector<double>> magnetic_susceptibilities;
    std::vector<float> temperatures;
    std::vector<float> magnetic_fields;

    std::vector<float> critical_temperatures;
    std::vector<int> critical_indices;
    std::vector<float> beta_exponents;

    int n;
    int iterations;
    float hMin;
    float hMax;
    int numHSteps;
    float hStep;
    float tempMin;
    float tempMax;
    int numTempSteps;
    float tempStep;

    Simulation(int n, int iterations, float hMin, float hMax, int numHSteps, float tempMin, float tempMax, int numTempSteps) {
        this->n = n;
        this->iterations = iterations;
        this->hMin = hMin;
        this->hMax = hMax;
        this->numHSteps = numHSteps;
        this->tempMin = tempMin;
        this->tempMax = tempMax;
        this->numTempSteps = numTempSteps;
        hStep = (hMax - hMin) / (numHSteps - 1);
        tempStep = (tempMax - tempMin) / (numTempSteps - 1);

        // numHSteps = numHSteps + 1;

        temperatures.resize(numTempSteps);
        magnetic_fields.resize(numHSteps);
        magnetic_fields[0] = hMin;

        for (int i = 1; i < numHSteps; i++) {
            //if (magnetic_fields[i-1] > 0 && magnetic_fields[i] < 0) {
            //    magnetic_fields[i] = 0.0f;  // Ensure we include zero field
            //} else {
            //    magnetic_fields[i] = hMin + i * hStep;
            //}

            magnetic_fields[i] = hMin + i * hStep;
        }

        // make it so that there is at least one point at zero magnetic field if the range includes it
        if (hMin < 0 && hMax > 0) {
            int zeroIndex = static_cast<int>(-hMin / hStep);
            magnetic_fields[zeroIndex] = 0.0f;  // Ensure we include zero field
            std::cout << "Adjusted magnetic field at index " << zeroIndex << " to include zero field: " << magnetic_fields[zeroIndex] << std::endl;
        }

        for (int i = 0; i < numTempSteps; i++) {
            temperatures[i] = tempMin + i * tempStep;
        }

        avg_magnetizations.resize(numHSteps, std::vector<double>(numTempSteps));
        magnetic_susceptibilities.resize(numHSteps, std::vector<double>(numTempSteps));
        critical_temperatures.resize(numHSteps);
        critical_indices.resize(numHSteps);
        beta_exponents.resize(numHSteps);
    }

    void runSimulation() {
        // 1. Generate unique seeds for each thread and parameter combination

        std::random_device rd;
        std::mt19937 master_gen(rd());
        std::uniform_int_distribution<uint32_t> seed_dist;
        std::vector<std::vector<uint32_t>> thread_seeds(numHSteps, std::vector<uint32_t>(numTempSteps));

        for (int i = 0; i < numHSteps; i++) {
            for (int j = 0; j < numTempSteps; j++) {
                thread_seeds[i][j] = seed_dist(master_gen);
            }
        }

// 2. Launch the parallel sweep
#pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < numTempSteps; i++) {
            for (int j = 0; j < numHSteps; j++) {
                Material material(n, temperatures[i], magnetic_fields[j], iterations, 1, thread_seeds[j][i]);
                material.runSimulation();
                avg_magnetizations[j][i] = material.averageMagnetization;
                material.MagneticSusceptibility();
                magnetic_susceptibilities[j][i] = material.magneticSusceptibility;
                // if (j == 0) {
                //     std::cout << temperatures[i] << " " << magnetic_fields[j] << std::endl;
                // }
            }
        }
    }

    void findCriticalTemperatureAndCalculateBeta() {
// Analyze the results to find the critical temperature for each magnetic field
#pragma omp parallel for schedule(dynamic)
        for (int j = 0; j < numHSteps; j++) {
            double maxSusceptibility = -1.0;
            int criticalTempIndex = -1;
            for (int i = 0; i < numTempSteps; i++) {
                if (magnetic_susceptibilities[j][i] > maxSusceptibility) {
                    maxSusceptibility = magnetic_susceptibilities[j][i];
                    criticalTempIndex = i;
                }
            }
            critical_temperatures[j] = temperatures[criticalTempIndex];
            critical_indices[j] = criticalTempIndex;

            std::vector<double> magnetizationNearTc;
            std::vector<double> tempDiffs;

            int startIndex = std::max(0, critical_indices[j] - 40);  // Look at 10 points below Tc
            int endIndex = critical_indices[j];                      // Up to Tc
            // filter data points just below critical temperature; skip points where |m|~0 (log would be -inf)
            for (int i = startIndex; i < endIndex; i++) {
                double absM = std::abs(avg_magnetizations[j][i]);
                double dT = critical_temperatures[j] - temperatures[i];
                if (absM > 0.01 && dT > 0.0) {
                    magnetizationNearTc.push_back(absM);
                    tempDiffs.push_back(dT);
                }
            }

            if (magnetizationNearTc.size() < 2) {
                beta_exponents[j] = std::numeric_limits<float>::quiet_NaN();  // Not enough data to fit
                continue;
            }

            // Perform a log-log fit to find beta
            double sumLogM = 0.0, sumLogT = 0.0, sumLogT2 = 0.0, sumLogMT = 0.0;
            for (size_t k = 0; k < magnetizationNearTc.size(); k++) {
                double logM = log(std::abs(magnetizationNearTc[k]));
                double logT = log(tempDiffs[k]);
                sumLogM += logM;
                sumLogT += logT;
                sumLogT2 += logT * logT;
                sumLogMT += logM * logT;
            }

            double nPoints = magnetizationNearTc.size();
            double slope = (nPoints * sumLogMT - sumLogM * sumLogT) / (nPoints * sumLogT2 - sumLogT * sumLogT);
            beta_exponents[j] = slope;
        }
    }

    void runIsingSimulation() {
        runSimulation();
        findCriticalTemperatureAndCalculateBeta();
    }

    void saveResultsToNPZ(const std::string& filename) {
        // prepare all arrays for saving
        std::vector<double> avgMagnetizationFlat;
        std::vector<double> magneticSusceptibilityFlat;

        avgMagnetizationFlat.resize(numHSteps * numTempSteps);
        magneticSusceptibilityFlat.resize(numHSteps * numTempSteps);

#pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < numHSteps; i++) {
            for (int j = 0; j < numTempSteps; j++) {
                avgMagnetizationFlat[i * numTempSteps + j] = avg_magnetizations[i][j];
                magneticSusceptibilityFlat[i * numTempSteps + j] = magnetic_susceptibilities[i][j];
            }
        }

        cnpy::npz_save(filename, "avg_magnetization", avgMagnetizationFlat.data(), std::vector<size_t>{(size_t)numHSteps, (size_t)numTempSteps}, "w");
        cnpy::npz_save(filename, "magnetic_susceptibility", magneticSusceptibilityFlat.data(),
                       std::vector<size_t>{(size_t)numHSteps, (size_t)numTempSteps}, "a");
        cnpy::npz_save(filename, "temperatures", temperatures.data(), std::vector<size_t>{(size_t)numTempSteps}, "a");
        cnpy::npz_save(filename, "magnetic_fields", magnetic_fields.data(), std::vector<size_t>{(size_t)numHSteps}, "a");
        cnpy::npz_save(filename, "critical_temperatures", critical_temperatures.data(), std::vector<size_t>{(size_t)numHSteps}, "a");
        cnpy::npz_save(filename, "beta_exponents", beta_exponents.data(), std::vector<size_t>{(size_t)numHSteps}, "a");
    }

    void saveResultsToCSV(const std::string& filename) {
        std::ofstream file(filename);
        file << "Temperature,MagneticField,AverageMagnetization,MagneticSusceptibility,BetaExponent\n";

        std::vector<std::string> buffers(numTempSteps);

#pragma omp parallel for schedule(static)
        for (int i = 0; i < numTempSteps; i++) {
            std::string& buffer = buffers[i];
            buffer.reserve(numHSteps * 100);
            for (int j = 0; j < numHSteps; j++) {
                buffer += std::to_string(temperatures[i]) + "," + std::to_string(magnetic_fields[j]) + "," +
                          std::to_string(avg_magnetizations[j][i]) + "," + std::to_string(magnetic_susceptibilities[j][i]) + "," + std::to_string(beta_exponents[j]) + "\n";
            }
        }

        for (int i = 0; i < numTempSteps; i++) {
            file << buffers[i];
        }
        file.close();
    }

    void saveResults() {
        std::filesystem::create_directory("output");
        saveResultsToNPZ("output/ising_results.npz");
        saveResultsToCSV("output/ising_results.csv");
    }
};

#endif  // PROCESSING_H