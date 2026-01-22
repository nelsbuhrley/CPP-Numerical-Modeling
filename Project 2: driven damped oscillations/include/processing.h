/**
 * @file processing.h
 * @brief Declares numerical integration methods and simulation control logic
 *
 * This header provides:
 * - integrator namespace: Contains RK4 and Euler-Cromer integration methods
 * - logic class: High-level simulation controller for user interaction and data management
 */

#pragma once

#include <functional>
#include <vector>

// TODO: Declare your data processing functions here.

/**
 * @namespace integrator
 * @brief Contains numerical integration algorithms for ODE systems
 *
 * Provides two integration methods optimized for different scenarios:
 * - RK4: High-accuracy 4th-order Runge-Kutta method
 * - Euler-Cromer: Semi-implicit method with good energy conservation for oscillators
 */
namespace integrator {
typedef std::vector<double> state_type;  ///< Type alias for state vectors

/**
 * @brief 4th-order Runge-Kutta (RK4) integration method
 *
 * A highly accurate explicit method for solving ordinary differential equations.
 * Uses four intermediate evaluations per step to achieve O(h^5) local truncation error.
 *
 * Algorithm per step:
 * - k1 = f(t_n, y_n)
 * - k2 = f(t_n + h/2, y_n + h*k1/2)
 * - k3 = f(t_n + h/2, y_n + h*k2/2)
 * - k4 = f(t_n + h, y_n + h*k3)
 * - y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
 *
 * @param state Initial state vector (modified in-place during integration)
 * @param derivatives Function that computes dy/dt given (y, dy/dt, t)
 * @param stopCondition Function returning true to continue, false to stop
 * @param timeStep Integration step size (smaller = more accurate but slower)
 * @return Trajectory of state vectors at each time step
 */
std::vector<state_type> rk4(state_type& state,
                            std::function<void(const state_type&, state_type&, double)> derivatives,
                            std::function<bool(const state_type&)> stopCondition, double timeStep);

/**
 * @brief Euler-Cromer (semi-implicit Euler) integration method
 *
 * A symplectic integrator that's particularly effective for oscillatory systems.
 * Updates velocity first, then uses the new velocity to update position.
 * This ordering preserves energy better than explicit Euler for Hamiltonian systems.
 *
 * Algorithm per step:
 * - v_{n+1} = v_n + a_n * h  (update velocity using current acceleration)
 * - x_{n+1} = x_n + v_{n+1} * h  (update position using NEW velocity)
 *
 * Note: For driven damped oscillators, "energy conservation" is less relevant,
 * but the method still performs well and is computationally efficient.
 *
 * @param state Initial state vector (modified in-place during integration)
 * @param derivatives Function that computes dy/dt given (y, dy/dt, t)
 * @param stopCondition Function returning true to continue, false to stop
 * @param timeStep Integration step size (can use larger steps than RK4)
 * @return Trajectory of state vectors at each time step
 */
std::vector<state_type> euler_chromer(
    state_type& state, std::function<void(const state_type&, state_type&, double)> derivatives,
    std::function<bool(const state_type&)> stopCondition, double timeStep);
}  // namespace integrator

/**
 * @class logic
 * @brief High-level controller for oscillator simulations
 *
 * Manages the complete simulation workflow:
 * - User interaction and input
 * - Selection between validation and custom modes
 * - Numerical integration execution
 * - Data export to CSV format
 * - Automated plotting via Python scripts
 */
class logic {
   public:
    /**
     * @brief Main entry point for the simulation program
     *
     * Presents a menu to the user for selecting between validation test
     * (predefined parameters) or custom simulation (user-specified parameters).
     * Calls the appropriate simulation method based on user choice.
     */
    void run();

    /**
     * @brief Exports simulation results to CSV and triggers plotting
     *
     * Generates a uniquely numbered output file, writes state trajectory data,
     * and calls a Python plotting script to visualize the results. The filename
     * is automatically incremented to avoid overwriting existing files.
     *
     * @param path Trajectory data (vector of state vectors)
     * @param filename Base filename for output (e.g., "Output/oscillator_output.csv")
     * @param maxPlotTime Maximum time to display in plots (seconds)
     * @param driverFrequency Driving frequency for Poincaré section analysis (rad/s)
     * @param dataInfo Optional metadata string to prepend to CSV file (default: "")
     */
    void outputResults(const std::vector<std::vector<double>>& path, const std::string& filename,
                       double maxPlotTime, double driverFrequency, std::string dataInfo = "");

    /**
     * @brief Runs a validation test with predefined parameters
     *
     * Creates a testOscillator instance with standard parameters and simulates
     * 72000 seconds of motion using Euler-Chromer integration. Results are
     * automatically exported and plotted.
     */
    void runValidationTest();

    /**
     * @brief Runs a custom simulation with user-specified parameters
     *
     * Prompts the user to input all oscillator parameters, time step, and
     * simulation duration. Uses RK4 integration for high accuracy. Results
     * are automatically exported and plotted.
     */
    void runCustomSimulation();
};