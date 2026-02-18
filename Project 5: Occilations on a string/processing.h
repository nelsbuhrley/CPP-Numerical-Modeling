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
#include "kiss_fft.h"

#ifndef PROCESSING_H
#define PROCESSING_H

/**
 *File Header
 *
 *Author: Nels Buhrley
 *Date: 2024-06-01
 *Description:
 *This header defines the `string` class which simulates oscillations on a string using a finite
 * difference method. It allows for superimposing Gaussian disturbances, sine waves, and natural
 * modes. The results can be output to CSV and NPZ files for further analysis and visualization. The
 * class uses OpenMP for parallelization and the kissfft library for FFT computations. The cnpy
 * library is used for saving results in NPZ format.
 *
 */

class string {
    double length;
    double waveSpeed;
    int segments;
    double stiffness;
    bool endIsFixed;
    double r;
    double timeStep;
    // double totalTime;
    int timeSteps;

    double stepSize;
    double dampFactor;

    std::vector<std::vector<double>> u;  // Displacement over time array

    // FFT results: [spatial_point][frequency_bin] -> magnitude
    std::vector<std::vector<double>> fftMagnitudes;
    // Frequency axis values (Hz)
    std::vector<double> fftFrequencies;

    std::vector<double> meanPowerSpectrum;  // Average power spectrum across all spatial points

   public:
    string(double length, double waveSpeed, int segments, double stiffness, double damping,
           bool endIsFixed, double totalTime)
        : length(length),
          segments(segments),
          stiffness(stiffness),
          endIsFixed(endIsFixed),
          r(0.95 / std::sqrt(1.0 + 4.0 * stiffness * segments * segments)),
          timeStep(r * length / segments / waveSpeed),
          timeSteps(static_cast<int>(totalTime / timeStep)),
          stepSize(length / segments),
          dampFactor(damping * timeStep),
          u(timeSteps, std::vector<double>(segments, 0.0)) {}

    /**
     * Superimpose a Gaussian disturbance on the initial condition.
     * @param center: Center of the Gaussian (0 to 1, as a fraction of string length)
     * @param width: Standard deviation of the Gaussian (as a fraction of string length)
     * @param amplitude: Peak amplitude of the disturbance
     */
    void superemposeGaussian(double center, double width, double amplitude) {
#pragma omp parallel for schedule(static)
        for (int i = 0; i < segments; i++) {
            double x = (i * stepSize) - center;
            u[0][i] += amplitude * std::exp(-(x * x) / (2 * width * width));
        }
    }

    /**
     * Superimpose a sine wave on the initial condition.
     * @param frequency: Frequency of the sine wave (Hz)
     * @param amplitude: Amplitude of the sine wave
     */
    void superemposeSine(double frequency, double amplitude) {
        // Superimpose a sine wave of given frequency and amplitude
#pragma omp parallel for schedule(static)
        for (int i = 0; i < segments; i++) {
            double x = (i * stepSize);
            u[0][i] += amplitude * std::sin(2 * M_PI * frequency * x / length);
        }
    }
    /**
     * Superimpose a natural mode on the initial condition.
     * @param modeNumber: The mode number (1 for fundamental, 2 for second harmonic, etc.)
     * @param amplitude: Amplitude of the natural mode
     */
    void superemposeNaturalMode(int modeNumber, double amplitude) {
        // Calculate the frequency of the natural mode and superimpose it
        double modeFrequency = (modeNumber) / (2 * length);
        superemposeSine(modeFrequency, amplitude);
    }
    /**
     * Simulate the string oscillations over time using a finite difference method.
     * The method updates the displacement array `u` in-place for each time step.
     * The update formula incorporates the wave equation with stiffness and damping terms.
     * The computation is parallelized over spatial points for efficiency.
     * The method handles fixed or free boundary conditions based on the `endIsFixed` flag. (Not Yet
     * Imolemented) The results are stored in the member variable `u` for later analysis or output.
     * Note: The first two time steps are initialized based on the initial conditions and the wave
     * equation, so the loop starts from t=1 and uses t-1 and t-2 for the update. The second time
     * term is handled carefully to avoid out-of-bounds access at the beginning of the simulation.
     *      The second space term is also handled with care to account for boundary conditions.
     *
     * Limatations and Complexities:
     * - The method assumes a uniform spatial grid and constant time step.
     * - The stability of the simulation depends on the choice of parameters, especially the time
     * step and stiffness.
     * - The method does not currently implement fixed boundary conditions, but it can be extended
     * to do so by modifying the update formula at the boundaries.
     * - The computational complexity is O(timeSteps * segments), which can be significant for large
     * simulations, but the use of OpenMP helps to mitigate this by parallelizing the spatial loop.
     */
    void simulate() {
        int last = endIsFixed ? segments - 1 : segments;

        for (int t = 1; t < timeSteps - 1; t++) {
            const std::vector<double>& secondTimeTerm = (t == 1) ? u[t - 1] : u[t - 2];
#pragma omp parallel for schedule(static)
            for (int i = 1; i < last; i++) {
                const double secondSpaceTermPlus =
                    (i < segments - 2)
                        ? u[t - 1][i + 2]
                        : -u[t - 1]
                            [i + 1];  // If end is fixed, mirror the value for the second space term
                const double secondSpaceTermMinus =
                    (i > 1)
                        ? u[t - 1][i - 2]
                        : -u[t - 1]
                            [i - 1];  // If end is fixed, mirror the value for the second space term
                // const double secondSpaceTermPlus = (i < segments - 2) ? u[t - 1][i + 2] : 0;  //
                // If end is fixed, mirror the value for the second space term const double
                // secondSpaceTermMinus = (i > 1) ? u[t - 1][i - 2] : 0;  // If end is fixed, mirror
                // the value for the second space term

                u[t][i] = ((2 - 2 * r * r - 6 * stiffness * r * r * segments * segments) *
                               u[t - 1][i]                         // First Time Term
                           - secondTimeTerm[i] * (1 - dampFactor)  // Second Time Term
                           + r * r * (1 + 4 * stiffness * segments * segments) *
                                 (u[t - 1][i + 1] + u[t - 1][i - 1])  // First Space Term
                           - stiffness * r * r * segments * segments *
                                 (secondSpaceTermPlus + secondSpaceTermMinus)) /
                          (1 + dampFactor);  // Second Space Term
            }
        }
    }

    /**
     * Compute the FFT magnitude spectrum for a given spatial point over time.
     * @param point: The spatial point index for which to compute the FFT (0 to segments-1)
     * @return A vector of magnitudes corresponding to the frequency bins of the FFT.
     */
    std::vector<double> FFTatPoint(int point) {
        // Number of time samples to transform. Use the next fast size for efficiency.
        int nfft = kiss_fft_next_fast_size(timeSteps);

        // Build complex input buffer (imag parts = 0)
        std::vector<kiss_fft_cpx> fin(nfft), fout(nfft);
        for (int t = 0; t < timeSteps; t++) {
            fin[t].r = static_cast<kiss_fft_scalar>(u[t][point]);
            fin[t].i = 0.0f;
        }
        // Zero-pad if nfft > timeSteps
        for (int t = timeSteps; t < nfft; t++) {
            fin[t].r = 0.0f;
            fin[t].i = 0.0f;
        }

        // Allocate kiss_fft configuration (forward FFT)
        kiss_fft_cfg cfg = kiss_fft_alloc(nfft, 0, nullptr, nullptr);
        if (!cfg) {
            std::cerr << "kiss_fft_alloc failed for nfft=" << nfft << std::endl;
            return {};
        }

        kiss_fft(cfg, fin.data(), fout.data());
        kiss_fft_free(cfg);

        // Only the first nfft/2+1 bins are unique for a real-valued signal
        int numBins = nfft / 2 + 1;
        std::vector<double> magnitudes(numBins);
        for (int k = 0; k < numBins; k++) {
            magnitudes[k] = std::sqrt(static_cast<double>(fout[k].r) * fout[k].r +
                                      static_cast<double>(fout[k].i) * fout[k].i);
        }

        // Populate frequency axis (Hz) once
        if (fftFrequencies.empty()) {
            fftFrequencies.resize(numBins);
            double sampleRate = 1.0 / timeStep;
            for (int k = 0; k < numBins; k++) {
                fftFrequencies[k] = k * sampleRate / nfft;
            }
        }

        // Store in member array (lazy allocation)
        if (static_cast<int>(fftMagnitudes.size()) != segments) {
            fftMagnitudes.assign(segments, std::vector<double>(numBins, 0.0));
        }
        fftMagnitudes[point] = magnitudes;

        return magnitudes;
    }

    /**
     * Compute the FFT magnitude spectra for all spatial points in parallel.
     * This method ensures that the shared member variables `fftFrequencies` and `fftMagnitudes`
     * are properly initialized before entering the parallel region to avoid data races.
     */
    void FFTallPoints() {
        // Pre-allocate shared members before entering the parallel region
        // to avoid data races on fftFrequencies and fftMagnitudes.
        if (fftFrequencies.empty()) {
            // Run one serial FFT to initialise fftFrequencies and size fftMagnitudes.
            FFTatPoint(0);
        }
        if (static_cast<int>(fftMagnitudes.size()) != segments) {
            fftMagnitudes.assign(segments, std::vector<double>(fftFrequencies.size(), 0.0));
        }
#pragma omp parallel for schedule(static)
        for (int i = 0; i < segments; i++) {
            FFTatPoint(i);
        }
    }

    /**
     * Compute the mean power spectrum across all spatial points.
     * This method should be called after `FFTallPoints()` to ensure that `fftMagnitudes` and
     * `fftFrequencies` are populated. The mean power spectrum is calculated by averaging the power
     * (magnitude squared) across all spatial points for each frequency bin. The results are stored
     * in the member variable `meanPowerSpectrum` for later output or analysis. Note: The method
     * assumes that `fftMagnitudes` is a 2D vector where the first dimension corresponds to spatial
     * points and the second dimension corresponds to frequency bins. The method iterates over each
     * frequency bin and accumulates the power across all spatial points, then divides by the number
     * of segments to get the mean power for that frequency bin. The computational complexity is
     * O(segments * numBins), which is efficient for typical values of segments and frequency bins.
     * The method does not use parallelization for the accumulation step to avoid race conditions on
     * the `meanPowerSpectrum` vector.
     */
    void computeMeanPowerSpectrum() {
        if (fftMagnitudes.empty()) {
            FFTallPoints();
        }
        int numBins = static_cast<int>(fftFrequencies.size());
        meanPowerSpectrum.assign(numBins, 0.0);
        // Accumulate power² per bin serially to avoid race conditions,
        // then divide to get the mean power at each frequency.
        for (int k = 0; k < numBins; k++) {
            double sumPower = 0.0;
            for (int i = 0; i < segments; i++) {
                sumPower += fftMagnitudes[i][k] * fftMagnitudes[i][k];  // Power = magnitude²
            }
            meanPowerSpectrum[k] = sumPower / segments;
        }
    }

    /**
     * Output the FFT magnitude spectra for all spatial points to a CSV file.
     * The CSV file will have a header row indicating the frequency bins and subsequent rows for
     * each spatial point. The first column will indicate the spatial position (x), and the
     * subsequent columns will contain the magnitude values for each frequency bin.
     * The method checks if the FFT results are already computed and calls `FFTallPoints()` if not.
     * It then builds a CSV string in memory to minimize file I/O overhead, and writes the entire
     * string to the specified file. The method handles file opening and closing, and includes error
     * handling for file I/O operations.
     *
     * Note: The method assumes that `fftFrequencies` and `fftMagnitudes` are properly populated
     * before it is called. The computational complexity is O(segments * numBins) due to the need to
     * iterate over all spatial points and frequency bins to build the output string. The method
     * does not use parallelization for building the output string to avoid complications with
     * string concatenation, but this could be optimized if needed by building separate strings for
     * each spatial point in parallel and then concatenating them at the end.
     */
    void outputFFTCSV(const std::string& filename) {
        if (fftMagnitudes.empty() || fftFrequencies.empty()) {
            FFTallPoints();
        }

        std::string output;
        int numBins = static_cast<int>(fftFrequencies.size());
        output.reserve(segments * numBins * 15);

        // Header: x position → frequency bins
        output += "Rows: Spatial Positions, Columns: Frequency (Hz)\n";
        output += "x \\ f";
        for (int k = 0; k < numBins; k++) {
            output += "," + std::to_string(fftFrequencies[k]);
        }
        output += "\n";

        // Data rows
        for (int i = 0; i < segments; i++) {
            output += std::to_string(i * stepSize);
            for (int k = 0; k < numBins; k++) {
                output += "," + std::to_string(fftMagnitudes[i][k]);
            }
            output += "\n";
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file for FFT output: " << filename << std::endl;
            return;
        }
        file << output;
        file.close();
    }
    /**
     * Output the time evolution of the string's displacement to a CSV file.
     * The CSV file will have a header row indicating the spatial positions and subsequent rows for
     * each time step. The first column will indicate the time step (t), and the subsequent columns
     * will contain the displacement values for each spatial point at that time step. The method
     * builds the CSV content in memory to minimize file I/O overhead, and writes the entire string
     * to the specified file at once.
     */
    void outputPositionResultsCSV(const std::string& filename) {
        // cash results to a string first to minimize file I/O overhead
        std::string output;
        output.reserve(timeSteps * segments * 20);  // rough estimate to minimize reallocations
        output += "Rows: Time Steps, Columns: Positions\n";
        output += "x → t ↓";
        for (int i = 0; i < segments; i++) {
            output += "," + std::to_string(i * stepSize);
        }
        output += "\n";

        // Each thread builds its own row string independently, then
        // the rows are concatenated in order after the parallel region.
        std::vector<std::string> rows(timeSteps);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < timeSteps; t++) {
            std::string row;
            row.reserve(segments * 20);
            row += "t" + std::to_string(t) + ",";
            for (int i = 0; i < segments; i++) {
                row += std::to_string(u[t][i]);
                if (i < segments - 1) {
                    row += ",";
                }
            }
            row += "\n";
            rows[t] = std::move(row);
        }
        for (int t = 0; t < timeSteps; t++) {
            output += rows[t];
        }
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file for output: " << filename << std::endl;
            return;
        }
        file << output;
        file.close();
    }
    /**
     * Output the mean power spectrum to a CSV file.
     * The CSV file will have two columns: "Frequency (Hz)" and "Mean Power".
     * The method checks if the mean power spectrum is already computed and calls
     * `computeMeanPowerSpectrum()` if not. It builds the CSV content in memory to minimize file I/O
     * overhead, and writes the entire string to the specified file at once. The method handles file
     * opening and closing, and includes error handling for file I/O operations.
     */
    void outputSpectralResultsCSV(const std::string& filename) {
        if (meanPowerSpectrum.empty()) {
            computeMeanPowerSpectrum();
        }

        std::string output;
        output.reserve(meanPowerSpectrum.size() * 20);
        output += "Frequency (Hz),Mean Power\n";
        for (size_t k = 0; k < meanPowerSpectrum.size(); k++) {
            output += std::to_string(fftFrequencies[k]) + "," +
                      std::to_string(meanPowerSpectrum[k]) + "\n";
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file for spectral output: " << filename << std::endl;
            return;
        }
        file << output;
        file.close();
    }
    /**
     * Output all results (positions and spectral data) to CSV files with a common filename prefix.
     * This method generates two CSV files: one for the time evolution of the string's displacement
     * and one for the mean power spectrum. The filenames are constructed by appending
     * "_positions.csv" and
     * "_spectrum.csv" to the provided filename prefix. The method calls
     * `outputPositionResultsCSV()` and `outputSpectralResultsCSV()` to generate the respective CSV
     * files. This method provides a convenient way to save all relevant results with a consistent
     * naming scheme, making it easier to manage and analyze the output data. The method does not
     * perform any additional computations and relies on the existing output methods to handle the
     * file writing and formatting.
     */
    void outputResultsCSV(const std::string& filename) {
        std::string positionFilename = filename + "_positions.csv";
        std::string spectralFilename = filename + "_spectrum.csv";
        outputPositionResultsCSV(positionFilename);
        outputSpectralResultsCSV(spectralFilename);
    }
    /**
     * Output all results (parameters, positions, frequencies, FFT magnitudes, and mean power
     * spectrum) to a single NPZ file. This method uses the cnpy library to save multiple arrays
     * into a single NPZ file. The method saves the simulation parameters, the time evolution of the
     * string's displacement, the frequency axis values, the FFT magnitude spectra for all spatial
     * points, and the mean power spectrum. The method checks if the FFT results and mean power
     * spectrum are already computed and calls the respective methods if not. The data is flattened
     * into 1D arrays where necessary to fit the expected input format for cnpy. The method handles
     * file writing and includes error handling for any issues that may arise during the saving
     * process. This approach allows for efficient storage and easy loading of the results in Python
     * or other environments that support NPZ files, facilitating further analysis and
     * visualization.
     */
    void outputResultsNPZ(const std::string& filename) {
        std::vector<double> parameters{length, waveSpeed, stiffness, r, timeStep};
        cnpy::npz_save(filename, "parameters", parameters.data(), {parameters.size()}, "w");
        // --- Position data: u[timeSteps][segments] ---
        std::vector<double> flatPositions;
        flatPositions.reserve(timeSteps * segments);
        for (const auto& row : u) {
            flatPositions.insert(flatPositions.end(), row.begin(), row.end());
        }
        cnpy::npz_save(filename, "positions", flatPositions.data(),
                       {static_cast<size_t>(timeSteps), static_cast<size_t>(segments)}, "a");

        // --- Frequency axis ---
        if (fftFrequencies.empty())
            FFTatPoint(0);
        cnpy::npz_save(filename, "frequencies", fftFrequencies.data(), {fftFrequencies.size()},
                       "a");

        // --- Full FFT magnitude spectra: fftMagnitudes[segments][numBins] ---
        if (fftMagnitudes.empty())
            FFTallPoints();
        int numBins = static_cast<int>(fftFrequencies.size());
        std::vector<double> flatFFT;
        flatFFT.reserve(segments * numBins);
        for (const auto& row : fftMagnitudes) {
            flatFFT.insert(flatFFT.end(), row.begin(), row.end());
        }
        cnpy::npz_save(filename, "fft_magnitudes", flatFFT.data(),
                       {static_cast<size_t>(segments), static_cast<size_t>(numBins)}, "a");

        // --- Mean power spectrum: meanPowerSpectrum[numBins] ---
        if (meanPowerSpectrum.empty())
            computeMeanPowerSpectrum();
        cnpy::npz_save(filename, "mean_power_spectrum", meanPowerSpectrum.data(),
                       {meanPowerSpectrum.size()}, "a");
    }
};

#endif