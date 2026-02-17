#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include "processing.h"

// test code that saves a 5 X 5 grid of zeros to a CSV file
int main() {
    // testString(length, waveSpeed, segments, stiffness, damping, endIsFixed, r, totalTime)
    string testString(1.0, 100.0, 200, 0, 0.001, true, 0.1, 1.0);
    // testString.superemposeGaussian(0.5, 0.1, 1.0);
    // testString.superemposeSine(5.0, 0.5);
    testString.superemposeNaturalMode(3, 0.5);
    testString.outputResultsCSV("string_oscillations1.csv");
    testString.simulate();
    testString.outputResultsCSV("string_oscillations2.csv");
    testString.outputResultsNPZ("string_oscillations.npz");

    return 0;
}