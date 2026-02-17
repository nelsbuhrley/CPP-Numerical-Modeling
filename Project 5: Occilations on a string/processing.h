#define _USE_MATH_DEFINES
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cnpy.h"

#ifndef PROCESSING_H
#define PROCESSING_H

class string {
    double length;
    double waveSpeed;
    int segments;
    double stiffness;
    double damping;
    bool endIsFixed;

    double timeStep;
    double totalTime;
    int timeSteps;


    double r;
    double stepSize;

     std::vector<std::vector<double>> u;  // Displacement over time array


   public:
    string(double length, double waveSpeed, int segments, double stiffness, double damping,
           bool endIsFixed, double totalTime)
        : length(length),
          waveSpeed(waveSpeed),
          segments(segments),
          stiffness(stiffness),
          damping(damping),
          endIsFixed(endIsFixed),
          r(0.95 / std::sqrt(1.0 + 4.0 * stiffness * segments * segments)),
          timeStep(r * length / segments / waveSpeed),
          totalTime(totalTime),
          timeSteps(static_cast<int>(totalTime / timeStep)),
          stepSize(length / segments),
          u(timeSteps, std::vector<double>(segments, 0.0)) {}

    void superemposeGaussian(double center, double width, double amplitude) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < segments; i++) {
            double x = (i * stepSize) - center;
            u[0][i] += amplitude * std::exp(-(x * x) / (2 * width * width));
        }
    }

    void superemposeSine(double frequency, double amplitude) {
        // Superimpose a sine wave of given frequency and amplitude
#pragma omp parallel for schedule(static)
        for (int i = 0; i < segments; i++) {
            double x = (i * stepSize);
            u[0][i] += amplitude * std::sin(2 * M_PI * frequency * x / length);
        }
    }

    void superemposeNaturalMode(int modeNumber, double amplitude) {
        // Calculate the frequency of the natural mode and superimpose it
        double modeFrequency = (modeNumber) / (2 * length);
        superemposeSine(modeFrequency, amplitude);
    }

    void simulate() {

        int last = endIsFixed ? segments - 1 : segments;

        for (int t = 1; t < timeSteps - 1; t++) {
            const std::vector<double>& secondTimeTerm = (t == 1) ? u[t - 1] : u[t - 2];
#pragma omp parallel for schedule(static)
            for (int i = 1; i < last; i++) {
                const double secondSpaceTermPlus = (i < segments - 2) ? u[t - 1][i + 2] : -u[t - 1][i + 1];  // If end is fixed, mirror the value for the second space term
                const double secondSpaceTermMinus = (i > 1) ? u[t - 1][i - 2] : -u[t - 1][i - 1];  // If end is fixed, mirror the value for the second space term
                // const double secondSpaceTermPlus = (i < segments - 2) ? u[t - 1][i + 2] : 0;  // If end is fixed, mirror the value for the second space term
                // const double secondSpaceTermMinus = (i > 1) ? u[t - 1][i - 2] : 0;  // If end is fixed, mirror the value for the second space term

                u[t][i] = (2 - 2 * r * r - 6 * stiffness * r * r * segments * segments) *
                              u[t - 1][i]  // First Time Term
                          - secondTimeTerm[i]    // Second Time Term
                          + r * r * (1 + 4 * stiffness * segments * segments) *
                                (u[t - 1][i + 1] + u[t - 1][i - 1])  // First Space Term
                          - stiffness * r * r * segments * segments *
                                (secondSpaceTermPlus + secondSpaceTermMinus);  // Second Space Term
            }
        }
    }

    void outputResultsCSV(const std::string& filename) {
        // cash results to a string first to minimize file I/O overhead
        std::string output;
        output.reserve(timeSteps * segments * 20);  // rough estimate to minimize reallocations
        output += "Rows: Time Steps, Columns: Posisions\n";
        output += "x → t ↓";
        for (int i = 0; i < segments; i++) {
            output += "," + std::to_string(i * stepSize);
        }
        output += "\n";
        for (int t = 0; t < timeSteps; t++) {
            output += "t" + std::to_string(t) + ",";
            for (int i = 0; i < segments; i++) {
                output += std::to_string(u[t][i]);
                if (i < segments - 1) {
                    output += ",";
                }
            }
            output += "\n";
        }
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file for output: " << filename << std::endl;
            return;
        }
        file << output;
        file.close();
    }

    void outputResultsNPZ(const std::string& filename) {
        // Convert 2D vector to 1D array for cnpy
        std::vector<double> flatData;
        flatData.reserve(timeSteps * segments);
        for (const auto& row : u) {
            flatData.insert(flatData.end(), row.begin(), row.end());
        };
        cnpy::npy_save(filename, flatData.data(), {static_cast<unsigned long>(timeSteps), static_cast<unsigned long>(segments)}, "w");
    }
};

#endif