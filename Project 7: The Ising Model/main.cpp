#include <iostream>
#include <vector>
#include <random>
#include <omp.h>
#include "processing.h"

int main() {
    int N = 100;
    int iterations = 200;

    // 1. Define your temperatures
    float minTemp = 0;
    float maxTemp = 45;
    int tempSteps = 200;

    // 2. Define your magnetic field range
    float hmin = -15.0;
    float hmax = 15.0;
    int numHSteps = 200;

    // 2. Run the simulation across the temperature range
    // parameters: lattice size, iterations, hMin, hMax, numHSteps, tempMin, tempMax, numTempSteps
    runIsingSimulation(N, iterations, hmin, hmax, numHSteps, minTemp, maxTemp, tempSteps);

    return 0;
}