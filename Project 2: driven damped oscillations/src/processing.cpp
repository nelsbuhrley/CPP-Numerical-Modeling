#include "processing.h"

#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "oscillator.h"

namespace integrator {
typedef std::vector<double> state_type;

std::vector<state_type> rk4(
    state_type& state,                                                        //
    std::function<void(const state_type&, state_type&, double)> derivatives,  //
    std::function<bool(const state_type&)> stopCondition,                     //
    double timeStep) {
    //
    std::vector<state_type> trajectory;
    trajectory.push_back(state);

    state_type k1(state.size());
    state_type k2(state.size());
    state_type k3(state.size());
    state_type k4(state.size());
    state_type tempState(state.size());

    while (stopCondition(state)) {
        // k1
        derivatives(state, k1, 0.0);

        // k2
        for (size_t i = 0; i < state.size(); ++i) {
            tempState[i] = state[i] + 0.5 * timeStep * k1[i];
        }
        derivatives(tempState, k2, 0.5 * timeStep);

        // k3
        for (size_t i = 0; i < state.size(); ++i) {
            tempState[i] = state[i] + 0.5 * timeStep * k2[i];
        }
        derivatives(tempState, k3, 0.5 * timeStep);

        // k4
        for (size_t i = 0; i < state.size(); ++i) {
            tempState[i] = state[i] + timeStep * k3[i];
        }
        derivatives(tempState, k4, timeStep);

        // Update state
        for (size_t i = 0; i < state.size(); ++i) {
            state[i] += (timeStep / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        trajectory.push_back(state);
        // std::cout << "Current time: " << state[0] << " seconds\r" << std::endl;
    }
    return trajectory;
}

std::vector<state_type> euler_chromer(
    state_type& state,                                                        //
    std::function<void(const state_type&, state_type&, double)> derivatives,  //
    std::function<bool(const state_type&)> stopCondition,                     //
    double timeStep) {
    //
    std::vector<state_type> trajectory;
    trajectory.push_back(state);

    state_type derivs(state.size());

    while (stopCondition(state)) {
        // Compute derivatives
        derivatives(state, derivs, 0.0);

        // Update angular velocity (state[2]) first
        state[2] += derivs[2] * timeStep;
        // Then update angle (state[1]) using the new angular velocity
        state[1] += state[2] * timeStep;
        // Update time (state[0])
        state[0] += timeStep;

        trajectory.push_back(state);
        // std::cout << "Current time: " << state[0] << " seconds\r" << std::endl;
    }
    return trajectory;
}
}  // namespace integrator

void logic::outputResults(const std::vector<std::vector<double>>& path, const std::string& filename,
                          double maxPlotTime, double driverFrequency) {
    int i = 0;
    std::string baseFilename = filename;
    std::string finalFilename = baseFilename.substr(0, baseFilename.find_last_of('.')) + "_" +
                                std::to_string(i) +
                                baseFilename.substr(baseFilename.find_last_of('.'));
    while (std::ifstream(finalFilename).good()) {
        finalFilename = baseFilename.substr(0, baseFilename.find_last_of('.')) + "_" +
                        std::to_string(i) + baseFilename.substr(baseFilename.find_last_of('.'));
        i++;
    }

    std::ofstream outFile(finalFilename);
    if (outFile.is_open()) {
        outFile << "Time,Angle,AngularVelocity\n";
        for (const auto& state : path) {
            outFile << state[0] << "," << state[1] << "," << state[2] << "\n";
        }
        outFile.close();
        std::cout << "Results written to " << finalFilename << std::endl;
    } else {
        std::cerr << "Error: Unable to open output file." << std::endl;
    }

    // call the ploting program

    std::string plotCommand = "python3 ploting.py " + finalFilename + " " +
                              std::to_string(maxPlotTime) + " " + std::to_string(driverFrequency);

    std::cout << "Executing plot command: " << plotCommand << std::endl;
    int plotResult = system(plotCommand.c_str());
    if (plotResult != 0) {
        std::cerr << "Error: Plotting command failed with code " << plotResult << std::endl;
    };
}

void logic::run() {
    std::cout << "Oscilation simulation program." << std::endl;
    std::cout << "Would you like to run the valadation test(1) or a custom simulation(2)? "
              << std::endl;
    int choice;
    std::cin >> choice;
    if (choice == 1) {
        // Run validation test
        std::cout << "Running validation test..." << std::endl;
        runValidationTest();
    } else if (choice == 2) {
        // Run custom simulation
        std::cout << "Running custom simulation..." << std::endl;
        runCustomSimulation();
    } else {
        std::cout << "Invalid choice. Exiting." << std::endl;
        return;
    }
}

void logic::runValidationTest() {
    testOscillator osc;

    osc.printParameters();

    auto derivFunc = osc.getDerivativeFunction();
    auto stopCondition = osc.getStopCondition(72000.0);

    double timeStep = 0.04;  // Time step for the simulation

    std::vector<double> initialState = osc.getState();
    std::vector<std::vector<double>> path =
        integrator::euler_chromer(initialState, derivFunc, stopCondition, timeStep);

    std::cout << "Simulation complete. Total steps: " << path.size() << std::endl;
    std::cout << "Initial state: Time = " << path.front()[0] << ", Angle = " << path.front()[1]
              << ", Angular Velocity = " << path.front()[2] << std::endl;
    std::cout << "Final state: Time = " << path.back()[0] << ", Angle = " << path.back()[1]
              << ", Angular Velocity = " << path.back()[2] << std::endl;
    outputResults(path, "Output/oscillator_output.csv", 180, osc.drivingFrequency);
};
//  oscillator(double mass_, double length_, double dampingCoefficient_, double initialAngle_,
//                double initialAngularVelocity_, double drivingForce_, double drivingFrequency_);

void logic::runCustomSimulation() {
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
    std::cout << "Enter time step (s): ";
    double timeStep;
    std::cin >> timeStep;
    std::cout << "Enter total simulation time (s): ";
    double totalTime;
    std::cin >> totalTime;
    std::cout << "Enter Plot max time (s): ";
    double maxPlotTime;
    std::cin >> maxPlotTime;

    oscillator osc(mass, length, dampingCoefficient, initialAngle, initialAngularVelocity,
                   drivingForce, drivingFrequency);
    osc.printParameters();
    auto derivFunc = osc.getDerivativeFunction();
    auto stopCondition = osc.getStopCondition(totalTime);

    std::vector<double> initialState = osc.getState();
    std::vector<std::vector<double>> path =
        integrator::rk4(initialState, derivFunc, stopCondition, timeStep);

    std::cout << "Simulation complete. Total steps: " << path.size() << std::endl;
    std::cout << "Initial state: Time = " << path.front()[0] << ", Angle = " << path.front()[1]
              << ", Angular Velocity = " << path.front()[2] << std::endl;
    std::cout << "Final state: Time = " << path.back()[0] << ", Angle = " << path.back()[1]
              << ", Angular Velocity = " << path.back()[2] << std::endl;
    outputResults(path, "Output/oscillator_output.csv", maxPlotTime, osc.drivingFrequency);
};
