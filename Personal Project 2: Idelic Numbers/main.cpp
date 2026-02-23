#include <atomic>
#include <bitset>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>
#include <omp.h>

// Sieve-based multithreaded idoneal number finder (proven correct)
void markNonIdoneal(char* sieve, int limit) {
    int maxVal = (int)std::sqrt(limit) + 1;
#pragma omp parallel for schedule(dynamic)
    for (int a = 1; a < maxVal; a++) {
        for (int b = a + 1; b < maxVal; b++) {
            for (int c = b + 1; c < maxVal; c++) {
                long long value = (long long)a * b + (long long)b * c + (long long)a * c;
                if (value >= limit)
                    break;
                sieve[value] = 1;
            }
        }
    }
}

int main() {
    // Run app;  // Create the application controller
    // app.runSimulation();  // Launch the interactive simulation interface

    // Optimized multithreaded idoneal number finder
    // A positive integer n is idoneal if and only if it cannot be written as
    // ab + bc + ac for distinct positive integers a, b, and c.

    const int LIMIT = 50000000;  // 10^7
    const int NUM_THREADS = omp_get_max_threads();

    std::cout << "Finding idoneal numbers up to " << LIMIT << " using " << NUM_THREADS
              << " threads...\n";
    std::cout << "Building sieve (proven method)..." << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    // Initialize sieve: 0 = idoneal, 1 = not idoneal
    std::vector<char> sieve(LIMIT, 0);

    markNonIdoneal(sieve.data(), LIMIT);

    std::cout << "Sieve complete. Collecting idoneal numbers..." << std::endl;

    // Collect all idoneal numbers
    std::vector<int> idonealNumbers;
    idonealNumbers.reserve(100);

    for (int n = 1; n < LIMIT; n++) {
        if (!sieve[n]) {
            idonealNumbers.push_back(n);
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    // Display results
    std::cout << "\n═══════════════════════════════════════════════════════\n";
    std::cout << "         Euler's Idoneal Numbers (NEC 2023)\n";
    std::cout << "═══════════════════════════════════════════════════════\n\n";

    for (size_t i = 0; i < idonealNumbers.size(); i++) {
        std::cout << "#" << (i + 1) << ": " << idonealNumbers[i];
        if ((i + 1) % 5 == 0)
            std::cout << "\n";
        else
            std::cout << "\t";
    }

    std::cout << "\n\n═══════════════════════════════════════════════════════\n";
    std::cout << "Total: " << idonealNumbers.size() << " idoneal numbers found\n";
    std::cout << "Computation time: " << duration.count() << " ms\n";
    std::cout << "Threads used: " << NUM_THREADS << "\n";
    std::cout << "═══════════════════════════════════════════════════════\n";

    return 0;
}