#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>

#include "processing.h"

/**
 * File Header
 *
 * Author: Nels Buhrley
 * Date: 2024-06-01
 * Description: This program simulates oscillations on a string using a finite difference method.
 *  It allows for superimposing Gaussian disturbances, sine waves, and natural modes.
 *  The results can be output to CSV and NPZ files for further analysis and visualization.
 * Note: The program uses OpenMP for parallelization and the kissfft library for FFT computations.
 *  The cnpy library is used for saving results in NPZ format.
 * 
 */

// test code that saves a 5 X 5 grid of zeros to a CSV file
int main() {
    // testString(length, waveSpeed, segments, stiffness, damping, endIsFixed, r, totalTime)
    string testString(1.0, 250.0, 100, 0.001, 10, true, 0.25);
    testString.superemposeGaussian(0.5, 0.1, 0.1);
    testString.superemposeGaussian(0.35, 0.05, 0.1);
    testString.superemposeGaussian(0.75, 0.03, -0.1);
    // testString.superemposeSine(5.0, 0.5);
    // testString.superemposeNaturalMode(3, 0.5);
    testString.simulate();
    testString.FFTallPoints();
    testString.computeMeanPowerSpectrum();

    testString.outputResultsCSV("string_oscillations");
    testString.outputResultsNPZ("string_oscillations.npz");

    return 0;
}