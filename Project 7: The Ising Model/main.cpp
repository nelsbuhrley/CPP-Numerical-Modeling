#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include "processing.h"

int main() {
    int N = 100;
    int iterations = 400;
    float h = 0.0;

    // 1. Define your temperatures
    float maxTemp = 1000;
    float minTemp = 0;
    float tempStep = 0.5;
    std::vector<float> temps;
    for (float T = minTemp; T <= maxTemp; T += tempStep) {
        temps.push_back(T);
    }

    // 2. The Master RNG (Runs on a single thread)
    std::random_device rd;
    std::mt19937 master_gen(rd());
    std::uniform_int_distribution<uint32_t> seed_dist;

    // 3. Generate a unique, perfectly random integer seed for every temperature
    std::vector<uint32_t> thread_seeds(temps.size());
    for (int i = 0; i < temps.size(); i++) {
        thread_seeds[i] = seed_dist(master_gen);
    }

    // 4. Launch the parallel sweep
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < temps.size(); i++) {
        // Pass the pre-generated seed safely into the constructor
        Material sim(N, temps[i], h, iterations, thread_seeds[i]);

        sim.runSimulation();

        // Output results for this temperature (e.g., save to file or print summary)
    }

    return 0;
}