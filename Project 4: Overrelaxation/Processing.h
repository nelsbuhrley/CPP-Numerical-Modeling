// u_new = u_old + omega*(u_relax - u_old) -> u_new = (1 - omega)*u_old + omega*u_relax
// #include <omp.h>
#define _USE_MATH_DEFINES
#include <omp.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "cnpy.h"

#ifndef PROCESSING_H
#define PROCESSING_H

// red-black ordering for 3D grid with OpenMP parallelization

void redOrBlackUpdate(double*** u, double*** f, int N, double omega, uint8_t evenOrOdd) {
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            int kStart = ((i + j) % 2 == evenOrOdd) ? 1 : 2;
            for (int k = kStart; k < N - 1; k += 2) {
                if (true) {
                    double u_relax = (1.0 / 6.0) * (u[i + 1][j][k] + u[i - 1][j][k] +
                                                    u[i][j + 1][k] + u[i][j - 1][k] +
                                                    u[i][j][k + 1] + u[i][j][k - 1] - f[i][j][k]);
                    u[i][j][k] = (1.0 - omega) * u[i][j][k] + omega * u_relax;
                } else {
                    u[i][j][k] = f[i][j][k];  // Dirichlet boundary condition
                }
            }
        }
    }
}

// Perform one complete red-black iteration
void fullStep(double*** u, double*** f, int N, double omega) {
    redOrBlackUpdate(u, f, N, omega, 0);  // Red update
    redOrBlackUpdate(u, f, N, omega, 1);  // Black update
}
// Helper function to allocate 3D array
double*** allocate3DArray(int N) {
    double*** array = new double**[N];
    for (int i = 0; i < N; i++) {
        array[i] = new double*[N];
        for (int j = 0; j < N; j++) {
            array[i][j] = new double[N];
        }
    }
    return array;
}

// Helper function to deallocate 3D array
void deallocate3DArray(double*** array, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            delete[] array[i][j];
        }
        delete[] array[i];
    }
    delete[] array;
}

double computeResidual(double*** u, double*** f, int N) {
    double residual_squared = 0.0;
#pragma omp parallel for reduction(+ : residual_squared) collapse(3) schedule(static)
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            for (int k = 1; k < N - 1; k++) {
                double laplacian = u[i + 1][j][k] + u[i - 1][j][k] + u[i][j + 1][k] +
                                   u[i][j - 1][k] + u[i][j][k + 1] + u[i][j][k - 1] -
                                   6.0 * u[i][j][k];
                double r = laplacian - f[i][j][k];
                residual_squared += r * r;
            }
        }
    }
    return std::sqrt(residual_squared);
}

class PotentialField {
   private:
    int N;                                       // Grid size
    [[maybe_unused]] double physicalDimensions;  // Physical size of the cube in meters
    double NDimensions;                          // Size of each grid cell in meters
    double*** u;                                 // Potential field
    double*** f;                                 // Source term (charge distribution)
    double omega;                                // Overrelaxation parameter

   public:
    PotentialField(int N, double physicalDimensions)
        : N(N), physicalDimensions(physicalDimensions) {
        u = allocate3DArray(N);
        f = allocate3DArray(N);
        omega = 2 / (1 + std::sqrt(2) + std::sin(3.14159 / N));  // Optimal omega for SOR
        NDimensions = physicalDimensions / N;
// Initialize u and f to zero
#pragma omp parallel for collapse(3) schedule(static)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    u[i][j][k] = 0.0;
                    f[i][j][k] = 0.0;
                }
            }
        }
    }

    ~PotentialField() {
        deallocate3DArray(u, N);
        deallocate3DArray(f, N);
    }

    void test_param() {
        // 10 cm radius dist centered at x=y=z=N/2 held at potential 0.25 V
        int center = N / 2;
        int radius = static_cast<int>(0.1 / NDimensions);
        int k = center;
        for (int i = center - radius; i <= center + radius; i++) {
            for (int j = center -
                         static_cast<int>(std::sqrt(radius * radius - (i - center) * (i - center)));
                 j <= center + static_cast<int>(
                                   std::sqrt(radius * radius - (i - center) * (i - center)));
                 j++) {
                f[i][j][k] = 0.25;
            }
        }
        // +1 uC point charge at (25,0,0) cm and (0,25,0) cm
        // -1 uC point charge at (-25,0,0) cm and (0,-25,0) cm
        center = N / 2;
        double chargeMagnitude = 1e-6;
        double pointVoltage = voltageOfPointCharge(chargeMagnitude) / 2000000;
        // define 1 cm radius sphere around point charge
        int pointRadius = static_cast<int>(0.01 / NDimensions);
        for (int i = center - pointRadius; i <= center + pointRadius; i++) {
            for (int j =
                     center - std::sqrt(pointRadius * pointRadius - (i - center) * (i - center));
                 j <= center + std::sqrt(pointRadius * pointRadius - (i - center) * (i - center));
                 j++) {
                for (int k = center -
                             std::sqrt(pointRadius * pointRadius - (i - center) * (i - center) -
                                       (j - center) * (j - center));
                     k <=
                     center + std::sqrt(pointRadius * pointRadius - (i - center) * (i - center) -
                                        (j - center) * (j - center));
                     k++) {
                    f[i + static_cast<int>(0.25 / NDimensions)][j][k] = pointVoltage;
                    f[i - static_cast<int>(0.25 / NDimensions)][j][k] = -pointVoltage;
                    f[i][j + static_cast<int>(0.25 / NDimensions)][k] = pointVoltage;
                    f[i][j - static_cast<int>(0.25 / NDimensions)][k] = -pointVoltage;
                }
            }
        }
    }

    double voltageOfPointCharge(double q) {
        return q / (4 * 3.14159 * 8.854e-12 * 0.01 / 2);  // V = q / (4πε₀r), r=(NDimensions/2)
    }

    void runSimulation(int maxIterations, double tolerance) {
        for (int iter = 0; iter < maxIterations; iter++) {
            fullStep(u, f, N, omega);
            if (iter % 100 == 0 || iter == maxIterations - 1) {
                double residual = computeResidual(u, f, N);
                std::cout << "Iteration " << iter << ", Residual: " << residual << std::endl;
                if (residual < tolerance) {
                    std::cout << "Converged after " << iter << " iterations." << std::endl;
                    break;
                }
                if (iter == maxIterations - 1) {
                    std::cout << "Reached maximum iterations." << std::endl;
                }
            }
        }
    }

    void outputResultsToCSV(const std::string& filename, std::string metadata) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file for output: " << filename << std::endl;
            return;
        }
        file << metadata << "\n";  // write metadata with # to start the line
        file << "# x,y,z,potential\n";
        std::vector<std::string> sections;
        int threadcount = omp_get_max_threads();
        sections.resize(threadcount);

#pragma omp parallel for schedule(static)
        for (int t = 0; t < threadcount; t++) {
            writeToSection(sections, t, N, threadcount);
        }

        // Write all sections to file serially
        for (const auto& section : sections) {
            file << section;
        }
        file.close();
    }

    void writeToSection(std::vector<std::string>& sections, int threadID, int N, int threadcount) {
        int kValuesPerSection = N / threadcount;
        int kStart = threadID * kValuesPerSection;
        int kEnd = (threadID == threadcount - 1) ? N : kStart + kValuesPerSection;
        std::string& section = sections[threadID];
        for (int k = kStart; k < kEnd; k++) {
            section += std::string("z") + std::to_string(k) + "\n";
            for (int j = 0; j < N; j++) {
                section += std::string("\t") + "y" + std::to_string(j) + "\n";
                for (int i = 0; i < N; i++) {
                    section +=
                        std::string("\t\t") + "x" + std::to_string(i) + "\t" + std::to_string(u[i][j][k]) + "\n";
                }
            }
        }
    }

    void exportTodotnpz(const std::string& filename) {
        // Flatten the 3D array into a 1D array for cnpy
        std::vector<double> flat_u;
        flat_u.reserve(N * N * N);

#pragma omp parallel for collapse(3) schedule(static)
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < N; j++) {
                for (int i = 0; i < N; i++) {
                    flat_u[k * N * N + j * N + i] = u[i][j][k];
                }
            }
        }

        // Save to .npz using cnpy
        cnpy::npz_save(filename, "potential", &flat_u[0],
                       {static_cast<unsigned long>(N), static_cast<unsigned long>(N),
                        static_cast<unsigned long>(N)},
                       "w");
    }
};

#endif