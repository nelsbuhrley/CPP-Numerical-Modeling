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
 *  Defines the Material class, which simulates the 3D Ising model on a cubic
 *      lattice using the Metropolis algorithm.
 * The class supports both random and uniform initial spin configurations,
 *      periodic boundary conditions, and precomputed energy tables for efficient updates.
 * The simulation can be run across a range of temperatures and magnetic fields,
 *      with results saved to NPZ files for analysis.
 *
 * error estamates and descussion:
 *
 * The Ising model has complexity OI(N^3 * iterations)
 * the simulation has complexity O(OI * numTempSteps * numHSteps)
 *
 * The main sources of error in the simulation are:
 * 1. Statistical fluctuations: The Monte Carlo method relies on random sampling,
 *      so results will vary between runs. Running more iterations can reduce this error.
 * 2. Finite size effects: A small lattice may not capture the true behavior of the system,
 *      especially near critical points. Increasing the lattice size can mitigate this.
 * 3. Discretization errors: The model is a simplified representation of reality,
 *      so it may not capture all physical phenomena accurately.
 *      However, it is a well-studied model that provides valuable insights into
 *      phase transitions and critical phenomena.
 */

class Material {
    int n;              // Size of the lattice (n x n x n)
    float temperature;  // Temperature of the system
    float h;            // Total magnetization of the system
    int numIterations;  // Number of iterations to run the simulation

    std::vector<int8_t> spins;  // 3D array: [x][y][z]

    float deltaE_table[2][7];  // Precomputed energy changes for spin flips
    float exp_table[2][7];     // Precomputed exp(-ΔE/T) values

    std::uniform_real_distribution<float> distribution;  // For random number generation
    std::mt19937 gen;                                    // Mersenne Twister RNG
    uint32_t seed;                                       // Seed for RNG

   public:
   /**
    * @brief Constructor for the Material class.
    * Initializes the lattice size, temperature, magnetization, number of iterations,
     * and random seed. It also sets up the RNG and precomputes energy tables for efficient updates.
     * The lattice is initialized with either random spins or a uniform spin value based on the constructor used.
    * @param n                Size of the lattice (n x n x n).
    * @param temperature      Temperature of the system.
    * @param magnetization    Total magnetization of the system (external magnetic field).
    * @param numIterations    Number of iterations to run the simulation.
    * @param seed             Seed for the random number generator to ensure reproducibility.
     * @param initialSpinValue Optional parameter to initialize all spins to a uniform value (+1 or -1).
     *                         If not provided, spins are initialized randomly.
    *
    * Note: The lattice size is set to n+2 to accommodate periodic boundary conditions, where the outer layers mirror the inner lattice.
     * The constructor also calls methods to initialize the spins and precompute energy tables for the Metropolis algorithm.
     * This setup allows for efficient simulation of the Ising model across a range of temperatures and magnetic fields.
    *
    */
    Material(int n, float temperature, float magnetization, int numIterations, uint32_t seed)
        : n(n + 2), temperature(temperature), h(magnetization), numIterations(numIterations), seed(seed) {
        establishRNG();
        initializeSpinsRandomly();
        precalculateEnergyTables();
    }

    Material(int n, float temperature, float magnetization, int numIterations, int8_t initialSpinValue, uint32_t seed)
        : n(n + 2), temperature(temperature), h(magnetization), numIterations(numIterations), seed(seed) {
        establishRNG();
        initializeSpinsUniformly(initialSpinValue);
        precalculateEnergyTables();
    }

    // Accessor and mutator for spins at specific lattice coordinates (x, y, z)
    inline int8_t getSpin(int x, int y, int z) {
        return spins[x * n * n + y * n + z];
    }

    inline void setSpin(int x, int y, int z, int8_t value) {
        spins[x * n * n + y * n + z] = value;
    }

    /**
     * @brief Initializes the spins of the lattice randomly to either +1 or -1.
     * Each spin is assigned a value based on a uniform distribution, ensuring an equal probability of being +1 or -1.
     * The method iterates through the inner lattice (excluding the boundary layers)
     */
    void initializeSpinsRandomly() {
        spins.resize(n * n * n);
        for (int x = 1; x < n - 1; x++) {
            for (int y = 1; y < n - 1; y++) {
                for (int z = 1; z < n - 1; z++) {
                    setSpin(x, y, z, (distribution(gen) < 0.5 ? -1 : 1));  // Randomly +1 or -1
                }
            }
        }
    }

    /**
     * @brief Initializes the spins of the lattice to a uniform value, either +1 or -1.
     * This method fills the entire inner lattice (excluding the boundary layers) with the specified spin value.
     * The spin value must be either +1 or -1; otherwise, an exception is thrown.
     *
     * @param spinValue The uniform spin value to initialize the lattice with (+1 or -1).
     */
    void initializeSpinsUniformly(int8_t spinValue) {
        if (spinValue != 1 && spinValue != -1) {
            throw std::invalid_argument("Spin value must be +1 or -1");
        }
        spins.resize(n * n * n, spinValue);
    }

    /**
     * @brief Precalculates the energy tables for efficient simulation.
     * This method precomputes the energy differences and corresponding exponential factors for all possible spin configurations.
     * The precomputed values are used during the simulation to avoid redundant calculations.
     */
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

    /**
     * @brief Establishes the random number generator for the simulation in a thread-safe manner.
     * This method initializes the Mersenne Twister RNG with a unique seed for each thread
     */
    void establishRNG() {
        // Thread-safe random number generator setup
        gen = std::mt19937(seed);
        distribution = std::uniform_real_distribution<float>(0.0, 1.0);
    }

    /**
     * @brief Attempts to flip a spin at the specified coordinates (x, y, z) based on the Metropolis algorithm.
     * The method calculates the energy change (ΔE) that would result from flipping
     * the spin and decides whether to flip it based on the precomputed energy tables and a random number.
     *
     * @param x The x-coordinate of the spin to potentially flip.
     * @param y The y-coordinate of the spin to potentially flip.
     * @param z The z-coordinate of the spin to potentially flip.
     *
     * The method first calculates the sum of the neighboring spins and maps it to an index for the energy tables.
     * It then retrieves the current spin state and uses the precomputed ΔE and exp(-ΔE/T) values to determine whether to flip the spin.
     * If the energy change is favorable (ΔE <= 0) or if a randomly generated number is less than exp(-ΔE/T), the spin is flipped by negating its current value.
     * This implementation allows for efficient updates during the simulation while adhering to the principles of the Metropolis algorithm for the Ising model
    */
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
        }
    }

    /**
     * @brief Performs one iteration of the Metropolis algorithm across the entire lattice.
     * The method consists of two main parts:
     * 1. Updating the boundary conditions: The outer layers of the lattice are updated to mirror the inner lattice, ensuring periodic boundary conditions.
     * 2. Iterating through the inner lattice: The method iterates through the inner lattice (excluding the boundary layers) in a Black and Red pattern and attempts to flip spins based on their coordinates.
     */
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

    /**
     * @brief Runs the full simulation for the specified number of iterations.
     * This method repeatedly calls the iteration() method, which performs one iteration of the Metropolis
     * algorithm across the entire lattice, for the total number of iterations specified in the constructor.
     * The method allows the simulation to evolve over time, enabling the system to reach equilibrium and
     */
    void runSimulation() {
        for (int i = 0; i < numIterations; i++) {
            iteration();
        }
    }
    /**
     * @brief Calculates the average magnetization of the system after the simulation has run.
     * The method iterates through the inner lattice (excluding the boundary layers) and sums the values of all spins to calculate the total magnetization. It then divides this total by the number of spins in the inner lattice to obtain the average magnetization per spin.
     * The average magnetization is a key observable in the Ising model, providing insights into the phase of the system (e.g., ferromagnetic or paramagnetic) and how it responds to changes in temperature and external magnetic
     */
    double getAverageMagnetization() {
        double totalMagnetization = 0.0;
        for (int x = 1; x < n - 1; x++) {
            for (int y = 1; y < n - 1; y++) {
                for (int z = 1; z < n - 1; z++) {
                    totalMagnetization += getSpin(x, y, z);
                }
            }
        }
        return totalMagnetization / ((n - 2) * (n - 2) * (n - 2));  // Average magnetization per spin
    }
};

/**
 * @brief Saves the simulation results to an NPZ file using the cnpy library.
 * This method takes the average magnetizations, temperatures, and magnetic field values and saves them in a structured format that can be easily loaded for analysis. The average magnetizations are flattened into a 1D array for storage, and the temperatures and magnetic field values are saved as separate arrays within the same NPZ file. This allows for efficient storage and retrieval of the simulation results for further analysis
 * @param avg_magnetizations A 2D vector containing the average magnetization values for each combination of temperature and magnetic field.
 * @param temps A vector containing the temperature values used in the simulation.
 * @param h_values A vector containing the magnetic field values used in the simulation.
 * @param filename The name of the NPZ file to save the results to.
 */
void saveResultsToNPZ(const std::vector<std::vector<double>>& avg_magnetizations, const std::vector<float>& temps, const std::vector<float>& h_values,
                      const std::string& filename) {
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

/**
 * @brief Runs the Ising model simulation across a range of temperatures and magnetic fields, and saves the results to an NPZ file.
 * This method generates the temperature and magnetic field ranges based on the specified minimum, maximum, and number of steps. It then initializes a master random number generator to create unique seeds for each thread in the parallel simulation. The simulation is run in parallel using OpenMP, where each thread simulates the Ising model for a specific combination of temperature and magnetic field. After the simulations are complete, the average magnetizations are saved to an NPZ file for analysis.
 * @param n The size of the lattice (n x n x n).
 * @param iterations The number of iterations to run the simulation for each combination of temperature and magnetic field.
 * @param hMin The minimum value of the magnetic field to simulate.
 * @param hMax The maximum value of the magnetic field to simulate.
 * @param numHSteps The number of magnetic field steps to simulate between hMin and hMax.
 * @param tempMin The minimum temperature to simulate.
 * @param tempMax The maximum temperature to simulate.
 * @param numTempSteps The number of temperature steps to simulate between tempMin and tempMax.
 */
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

#endif  // PROCESSING_H