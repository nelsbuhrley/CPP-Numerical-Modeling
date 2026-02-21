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
 *
 */


int main() {
    float movementRadius = 0.025f;  // 1 cm
    float cubeSize = 1.f;         // 1 m cube
    size_t steps = 1000;
    size_t numPoints = 2500;
    std::string outputFilename = "output/defusion_output";
    Space space(movementRadius, cubeSize, steps, numPoints);
    space.Propagate();
    space.calculateRMS();
    space.saveToCSV(outputFilename);
    space.saveToNPZ(outputFilename);
    return 0;
}
