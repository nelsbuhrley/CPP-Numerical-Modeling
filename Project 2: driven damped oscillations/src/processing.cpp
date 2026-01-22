/**
 * @file processing.cpp
 * @brief Implementation of numerical integration methods and simulation control logic
 *
 * Provides two numerical integrators (RK4 and Euler-Cromer) for solving ODEs,
 * plus high-level simulation management through the logic class.
 */

#include "processing.h"

#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "oscillator.h"

/**
 * @namespace integrator
 * @brief Contains numerical integration algorithms
 */
namespace integrator {
typedef std::vector<double> state_type;  ///< State vector type

/**
 * @brief Implements 4th-order Runge-Kutta integration
 *
 * RK4 is a highly accurate explicit method for solving ordinary differential equations.
 * It achieves O(h^4) global error by taking four intermediate function evaluations
 * per time step and computing a weighted average.
 *
 * @param state Initial state vector (modified in-place during integration)
 * @param derivatives Function computing dy/dt given (y, dy/dt, t)
 * @param stopCondition Predicate returning true to continue integration
 * @param timeStep Integration step size (h)
 * @return Complete trajectory as vector of state vectors
 */
std::vector<state_type> rk4(
    state_type& state,                                                        //
    std::function<void(const state_type&, state_type&, double)> derivatives,  //
    std::function<bool(const state_type&)> stopCondition,                     //
    double timeStep) {
    //
    std::vector<state_type> trajectory;
    trajectory.push_back(state);  // Store initial condition

    // Allocate storage for intermediate calculations
    state_type k1(state.size());         // Slope at beginning of interval
    state_type k2(state.size());         // Slope at midpoint using k1
    state_type k3(state.size());         // Slope at midpoint using k2
    state_type k4(state.size());         // Slope at end of interval
    state_type tempState(state.size());  // Temporary state for intermediate evaluations

    // Main integration loop - continue while stop condition is true
    while (stopCondition(state)) {
        // Step 1: Compute k1 = f(t_n, y_n) - slope at start of interval
        derivatives(state, k1, 0.0);

        // Step 2: Compute k2 = f(t_n + h/2, y_n + h*k1/2) - slope at midpoint using k1
        for (size_t i = 0; i < state.size(); ++i) {
            tempState[i] = state[i] + 0.5 * timeStep * k1[i];
        }
        derivatives(tempState, k2, 0.5 * timeStep);

        // Step 3: Compute k3 = f(t_n + h/2, y_n + h*k2/2) - slope at midpoint using k2
        for (size_t i = 0; i < state.size(); ++i) {
            tempState[i] = state[i] + 0.5 * timeStep * k2[i];
        }
        derivatives(tempState, k3, 0.5 * timeStep);

        // Step 4: Compute k4 = f(t_n + h, y_n + h*k3) - slope at end of interval
        for (size_t i = 0; i < state.size(); ++i) {
            tempState[i] = state[i] + timeStep * k3[i];
        }
        derivatives(tempState, k4, timeStep);

        // Update state using weighted average: y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
        // Weights: 1/6 for endpoints (k1, k4) and 2/6 for midpoints (k2, k3)
        for (size_t i = 0; i < state.size(); ++i) {
            state[i] += (timeStep / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        trajectory.push_back(state);  // Store updated state
        // std::cout << "Current time: " << state[0] << " seconds\r" << std::endl;
    }
    return trajectory;
}

// Limitations and error estimates for RK4:
// - Fixed time step can lead to instability if too large.
// - Global error is O(h^4), but local truncation error is O(h^5).
// - Not suitable for stiff equations without modification.
// - Does not conserve energy in Hamiltonian systems.

/**
 * @brief Implements Euler-Cromer (semi-implicit Euler) integration
 *
 * A symplectic integrator particularly suited for oscillatory systems.
 * Unlike explicit Euler, it updates velocity first, then uses the updated
 * velocity to compute the new position. This ordering provides better
 * energy conservation for Hamiltonian systems.
 *
 * For a state vector [time, position, velocity]:
 * 1. v_{n+1} = v_n + a_n * h  (update velocity using current acceleration)
 * 2. x_{n+1} = x_n + v_{n+1} * h  (update position using NEW velocity)
 * 3. t_{n+1} = t_n + h
 *
 * @param state Initial state vector (modified in-place during integration)
 * @param derivatives Function computing dy/dt given (y, dy/dt, t)
 * @param stopCondition Predicate returning true to continue integration
 * @param timeStep Integration step size (h)
 * @return Complete trajectory as vector of state vectors
 */
std::vector<state_type> euler_chromer(
    state_type& state,                                                        //
    std::function<void(const state_type&, state_type&, double)> derivatives,  //
    std::function<bool(const state_type&)> stopCondition,                     //
    double timeStep) {
    //
    std::vector<state_type> trajectory;
    trajectory.push_back(state);  // Store initial condition

    state_type derivs(state.size());  // Storage for derivatives

    // Main integration loop - continue while stop condition is true
    while (stopCondition(state)) {
        // Compute all derivatives at current state
        derivatives(state, derivs, 0.0);

        // CRITICAL: Update velocity (state[2]) FIRST using current acceleration
        // This is the "Cromer" modification that makes the method semi-implicit
        state[2] += derivs[2] * timeStep;

        // Then update position (state[1]) using the NEWLY COMPUTED velocity
        // This ordering preserves symplectic structure and improves stability
        state[1] += state[2] * timeStep;

        // Update time (state[0]) - always advances at constant rate
        state[0] += timeStep;

        trajectory.push_back(state);
        // std::cout << "Current time: " << state[0] << " seconds\r" << std::endl;
    }
    return trajectory;
}
}  // namespace integrator

// Limaitations and error estimates for Euler-Cromer:
// - Simple and easy to implement.
// - First-order method with global error O(h).
// - More stable than standard Euler for oscillatory systems.
// - Does not conserve energy in Hamiltonian systems over long times.
// - Not suitable for stiff equations.
// - Accuracy can be low even for small time steps.

/**
 * @brief Exports simulation data to CSV and triggers Python plotting
 *
 * Performs three main tasks:
 * 1. Generates a uniquely numbered filename to avoid overwriting existing data
 * 2. Writes trajectory data to CSV with optional metadata header
 * 3. Invokes Python plotting script with appropriate parameters
 *
 * File naming: If "Output/data.csv" exists, creates "Output/data_0.csv",
 * then "Output/data_1.csv", etc.
 *
 * @param path Trajectory data - vector of state vectors [time, angle, velocity]
 * @param filename Base filename (e.g., "Output/oscillator_output.csv")
 * @param maxPlotTime Maximum time to display in plots (for first 3 panels)
 * @param driverFrequency Driving frequency for Poincaré section sampling
 * @param dataInfo Optional metadata to prepend to file (default: empty)
 */
void logic::outputResults(const std::vector<std::vector<double>>& path, const std::string& filename,
                          double maxPlotTime, double driverFrequency, std::string dataInfo) {
    // Generate unique filename by incrementing counter until unused name is found
    int i = 0;
    std::string baseFilename = filename;

    // Construct initial candidate filename: "base_0.ext"
    std::string finalFilename = baseFilename.substr(0, baseFilename.find_last_of('.')) + "_" +
                                std::to_string(i) +
                                baseFilename.substr(baseFilename.find_last_of('.'));

    // Increment counter until we find a filename that doesn't exist
    while (std::ifstream(finalFilename).good()) {
        finalFilename = baseFilename.substr(0, baseFilename.find_last_of('.')) + "_" +
                        std::to_string(i) + baseFilename.substr(baseFilename.find_last_of('.'));
        i++;
    }

    // Open output file for writing
    std::ofstream outFile(finalFilename);
    if (outFile.is_open()) {
        // Write optional metadata header (typically comments prefixed with #)
        outFile << dataInfo << "\n";

        // Write CSV header row
        outFile << "Time,Angle,AngularVelocity\n";

        // Write trajectory data - one row per time step
        for (const auto& state : path) {
            outFile << state[0] << "," << state[1] << "," << state[2] << "\n";
        }
        outFile.close();
        std::cout << "Results written to " << finalFilename << std::endl;
    } else {
        std::cerr << "Error: Unable to open output file." << std::endl;
    }

    // Call the Python plotting program to generate visualizations
    // Arguments: <csv_file> <max_plot_time> <driver_frequency>
    // The plotting script will create 4-panel analysis figure

    std::string plotCommand = "python3 plotting.py " + finalFilename + " " +
                              std::to_string(maxPlotTime) + " " + std::to_string(driverFrequency);

    std::cout << "Executing plot command: " << plotCommand << std::endl;
    int plotResult = system(plotCommand.c_str());  // Execute system command
    if (plotResult != 0) {
        std::cerr << "Error: plotting command failed with code " << plotResult << std::endl;
    };
}

/**
 * @brief Main program entry point - presents menu and routes to simulation mode
 *
 * Provides interactive menu for user to select:
 * - Option 1: Validation test with predefined parameters
 * - Option 2: Custom simulation with user-specified parameters
 *
 * Invalid selections result in graceful exit.
 */
void logic::run() {
    std::cout << "Oscilation simulation program." << std::endl;
    std::cout << "Would you like to run the valadation test(1) or a custom simulation(2)? "
              << std::endl;

    int choice;
    std::cin >> choice;

    if (choice == 1) {
        // Run validation test with predefined parameters
        std::cout << "Running validation test..." << std::endl;
        runValidationTest();
    } else if (choice == 2) {
        // Run custom simulation with user-specified parameters
        std::cout << "Running custom simulation..." << std::endl;
        runCustomSimulation();
    } else {
        // Invalid input - exit gracefully
        std::cout << "Invalid choice. Exiting." << std::endl;
        return;
    }
}

/**
 * @brief Executes validation test with predefined standard parameters
 *
 * Uses testOscillator class which initializes with project specification:
 * - Mass: 1.0 kg
 * - Length: 9.8 m
 * - Damping: 0.5
 * - Initial angle: 0.2 rad
 * - Initial velocity: 0.0 rad/s
 * - Driving force: 1.2 N
 * - Driving frequency: 2/3 rad/s
 *
 * Simulates 72000 seconds (20 hours) using Euler-Cromer with 0.04s time step.
 * Results are exported with full metadata and plotted.
 */
void logic::runValidationTest() {
    // Create oscillator with standard validation parameters
    testOscillator osc;

    // Display parameters for verification
    osc.printParameters();

    // Get derivative function and stop condition as lambda wrappers
    auto derivFunc = osc.getDerivativeFunction();
    auto stopCondition = osc.getStopCondition(72000.0);  // Simulate 20 hours

    double timeStep = 0.04;  // Time step: 0.04s (25 steps per second)

    // Run Euler-Cromer integration
    std::vector<double> initialState = osc.getState();
    std::vector<std::vector<double>> path =
        integrator::euler_chromer(initialState, derivFunc, stopCondition, timeStep);

    std::cout << "Simulation complete. Total steps: " << path.size() << std::endl;
    std::cout << "Initial state: Time = " << path.front()[0] << ", Angle = " << path.front()[1]
              << ", Angular Velocity = " << path.front()[2] << std::endl;
    std::cout << "Final state: Time = " << path.back()[0] << ", Angle = " << path.back()[1]
              << ", Angular Velocity = " << path.back()[2] << std::endl;

    std::string dataInfo =                                                           //
        std::string("# Initial Parameters: ")                                        //
        + "\n#    Mass: " + std::to_string(osc.mass)                                 //
        + "\n#    Length: " + std::to_string(osc.length)                             //
        + "\n#    Damping Coefficient: " + std::to_string(osc.dampingCoefficient)    //
        + "\n#    Initial Angle: " + std::to_string(osc.angle)                       //
        + "\n#    Initial Angular Velocity: " + std::to_string(osc.angularVelocity)  //
        + "\n#    Driving Force: " + std::to_string(osc.drivingForce)                //
        + "\n#    Driving Frequency: " + std::to_string(osc.drivingFrequency)        //
        + "\n# Integration Information:"                                             //
        + "\n#    Time Step: " + std::to_string(timeStep)                            //
        + "\n#    Total Simulation Time: " + std::to_string(path.back()[0])          //
        + "\n#    Integration Method: Euler-Cromer"                                  //
        + "\n# Final State:"                                                         //
        + "\n#    Time = " + std::to_string(path.back()[0])                          //
        + "\n#    Angle = " + std::to_string(path.back()[1])                         //
        + "\n#    Angular Velocity = " + std::to_string(path.back()[2]);

    outputResults(path, "Output/oscillator_output.csv", 180, osc.drivingFrequency, dataInfo);
};
//  oscillator(double mass_, double length_, double dampingCoefficient_, double initialAngle_,
//                double initialAngularVelocity_, double drivingForce_, double drivingFrequency_);

/**
 * @brief Executes custom simulation with user-specified parameters
 *
 * Prompts user to input all physical parameters, integration settings,
 * and plotting parameters. Uses high-accuracy RK4 integration method.
 * Results are exported with metadata and plotted.
 *
 * User inputs:
 * - Physical parameters: mass, length, damping, forces
 * - Initial conditions: angle, angular velocity
 * - Integration: time step, total duration
 * - Plotting: maximum time for first 3 plots
 */
void logic::runCustomSimulation() {
    // Prompt for all physical parameters
    std::cout << "Enter mass (kg): ";
    double mass;
    std::cin >> mass;

    std::cout << "Enter length (m): ";
    double length;
    std::cin >> length;

    std::cout << "Enter damping coefficient: ";
    double dampingCoefficient;
    std::cin >> dampingCoefficient;

    std::cout << "Enter initial angle (rad): ";
    double initialAngle;
    std::cin >> initialAngle;

    std::cout << "Enter initial angular velocity (rad/s): ";
    double initialAngularVelocity;
    std::cin >> initialAngularVelocity;

    std::cout << "Enter driving force (N): ";
    double drivingForce;
    std::cin >> drivingForce;

    std::cout << "Enter driving frequency (rad/s): ";
    double drivingFrequency;
    std::cin >> drivingFrequency;

    // Prompt for integration parameters
    std::cout << "Enter time step (s): ";
    double timeStep;
    std::cin >> timeStep;

    std::cout << "Enter total simulation time (s): ";
    double totalTime;
    std::cin >> totalTime;

    std::cout << "Enter Plot max time (s): ";
    double maxPlotTime;
    std::cin >> maxPlotTime;

    // Create oscillator with user-specified parameters
    oscillator osc(mass, length, dampingCoefficient, initialAngle, initialAngularVelocity,
                   drivingForce, drivingFrequency);

    // Display parameters for verification
    osc.printParameters();

    // Get derivative function and stop condition as lambda wrappers
    auto derivFunc = osc.getDerivativeFunction();
    auto stopCondition = osc.getStopCondition(totalTime);

    // Run RK4 integration for high accuracy with user-specified time step
    std::vector<double> initialState = osc.getState();
    std::vector<std::vector<double>> path =
        integrator::rk4(initialState, derivFunc, stopCondition, timeStep);

    std::cout << "Simulation complete. Total steps: " << path.size() << std::endl;
    std::cout << "Initial state: Time = " << path.front()[0] << ", Angle = " << path.front()[1]
              << ", Angular Velocity = " << path.front()[2] << std::endl;
    std::cout << "Final state: Time = " << path.back()[0] << ", Angle = " << path.back()[1]
              << ", Angular Velocity = " << path.back()[2] << std::endl;

    std::string dataInfo =                                                           //
        std::string("# Initial Parameters: ")                                        //
        + "\n#    Mass: " + std::to_string(osc.mass)                                 //
        + "\n#    Length: " + std::to_string(osc.length)                             //
        + "\n#    Damping Coefficient: " + std::to_string(osc.dampingCoefficient)    //
        + "\n#    Initial Angle: " + std::to_string(osc.angle)                       //
        + "\n#    Initial Angular Velocity: " + std::to_string(osc.angularVelocity)  //
        + "\n#    Driving Force: " + std::to_string(osc.drivingForce)                //
        + "\n#    Driving Frequency: " + std::to_string(osc.drivingFrequency)        //
        + "\n# Integration Information:"                                             //
        + "\n#    Time Step: " + std::to_string(timeStep)                            //
        + "\n#    Total Simulation Time: " + std::to_string(path.back()[0])          //
        + "\n#    Integration Method: Euler-Cromer"                                  //
        + "\n# Final State:"                                                         //
        + "\n#    Time = " + std::to_string(path.back()[0])                          //
        + "\n#    Angle = " + std::to_string(path.back()[1])                         //
        + "\n#    Angular Velocity = " + std::to_string(path.back()[2]);

    outputResults(path, "Output/oscillator_output.csv", maxPlotTime, osc.drivingFrequency,
                  dataInfo);
};
