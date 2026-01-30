
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include "objects.h"
#include "processing.h"
using namespace std;

/**
 * @brief Main application controller for celestial dynamics simulations
 * Handles user interaction, simulation setup, execution, and output.
 * Utilizes CelestialSystem and integrator methods to run simulations.
 * Author: Nels Buhrley
 * Date: June 2024
 *
 * Usage:
 * - Create a Run instance and call runSimulation()
 * - Follow prompts to select simulation type and parameters
 * - Results are output to data files for analysis and visualization
 *
 * 
 */

class Run {
   public:
    void runSimulation() {
        // Example usage of the integrators can be placed here
    }
    Run() {
        cout << "Celestial Dynamics Simulation" << endl;
        cout << "-----------------------------" << endl;
        cout << "Would you like to run a verification simulation(1), the solar system(2), or a "
                "custom simulation(3)? (1/2/3): ";
        int choice;
        cin >> choice;
        CelestialSystem system;
        std::array<double, 2> integrationParameters;  // [0] timeStep, [1] totalTime
        int integratorChoice = 0;                     // 1=RK4, 2=Euler-Cromer, 3=Verlet
        std::string dataInfo = "";
        if (choice == 1) {
            cout << "Running verification simulation..." << endl;
            VerificationSimulation(system, integrationParameters, integratorChoice);

        } else if (choice == 2) {
            cout << "Running solar system simulation..." << endl;
            SolarSystemSimulation(system, integrationParameters, integratorChoice);
        } else if (choice == 3) {
            cout << "Running custom simulation..." << endl;
            CustomSimulation(system, integrationParameters, integratorChoice);
        } else {
            cout << "Invalid choice. Exiting." << endl;
        }
        std::string tempString;
        for (size_t i = 0; i < system.objects.size(); ++i) {
            tempString = "#Object " + std::to_string(i + 1) + " Info:\n" +
                         system.objects[i].getInfo() + "\n";
            dataInfo += tempString;
            std::cout << tempString;
        }

        dataInfo +=
            std::string("# Integration Parameters: ") +
            "\n#    Time Step: " + std::to_string(integrationParameters[0]) + " years\n" +
            "#    Total Time: " + std::to_string(integrationParameters[1]) + " years\n" +
            "#    Integrator: " +
            (integratorChoice == 1
                 ? "RK4"
                 : (integratorChoice == 2 ? "Euler-Cromer"
                                          : (integratorChoice == 3 ? "Verlet" : "Unknown"))) +
            "\n";
        // run the chosen simulation

        // Setup common integration components
        std::vector<double> state = system.getStateVector(0.0);
        auto derivatives = system.getDerivativeFunction();
        auto stopCondition = [totalTime = integrationParameters[1]](const std::vector<double>& s) {
            return s[0] < totalTime;
        };
        double timeStep = integrationParameters[0];
        std::vector<std::vector<double>> trajectory;
        cout << "Starting simulation..." << endl;
        cout << "Integrator choice: " << integratorChoice << endl;
        if (integratorChoice == 1) {
            trajectory =
                integrator::rk4<std::vector<double>>(state, derivatives, stopCondition, timeStep);
            cout << "RK4 simulation complete. Steps: " << trajectory.size() << endl;
        } else if (integratorChoice == 2) {
            trajectory = integrator::euler_chromer<std::vector<double>>(state, derivatives,
                                                                        stopCondition, timeStep);
            cout << "Euler-Cromer simulation complete. Steps: " << trajectory.size() << endl;
        } else if (integratorChoice == 3) {
            // Verlet uses position-only format
            auto positionState = system.getVerletStateVector(0.0);
            auto velocities = system.getVelocityVector();
            auto accelFunc = system.getVerletAccelerationFunction();
            auto verletStopCondition =
                [totalTime = integrationParameters[1]](const std::vector<double>& s) {
                    return s[0] < totalTime;
                };

            // Initialize previous state for Verlet
            auto previousState = integrator::initializeVerletPreviousState<std::vector<double>>(
                positionState, velocities, accelFunc, timeStep);

            trajectory = integrator::verlet<std::vector<double>>(
                positionState, previousState, accelFunc, verletStopCondition, timeStep);
            cout << "Verlet simulation complete. Steps: " << trajectory.size() << endl;
        } else {
            cout << "Invalid integrator choice. Exiting." << endl;
        }

        outputResults(trajectory, system, dataInfo);
    }

    void VerificationSimulation(CelestialSystem& system,
                                std::array<double, 2>& integrationParameters,
                                int& integratorChoice) {
        std::cout << "Choose from the simulation options below:" << std::endl;
        std::cout << "1. Two-body orbit (Earth-Sun)" << std::endl;
        std::cout << "2. Two-body orbit (Jupiter-Sun)" << std::endl;
        std::cout << "3. Two-body orbit (2 Suns 5.2 AU apart)" << std::endl;
        std::cout << "4. Three-body problem (Sun-Earth-Moon)" << std::endl;
        std::cout << "5. Three-body problem (Sun-Earth-Jupiter)" << std::endl;
        std::cout << "Enter your choice (1-5): ";
        int choice;
        std::cin >> choice;

        // Masses: Sun 1.0000 M⊙, Earth 3.0027⨉10-6 M⊙, Jupiter 9.54588⨉10-4 M⊙.
        // Initial positions relative to Sun: Earth (1, 0, 0) AU, Jupiter (-5.2, 0, 0) AU.
        // Initial velocities relative to Sun: Earth (0, 2π, 0) AU/y, Jupiter (0, -0.876897π, 0)
        // AU/y

        // With these initial conditions, let the system evolve for 11.2 years. Submit the
        // output files and trajectory plots for four cases:

        // Jupiter with its normal mass.
        // Jupiter with 10 times its normal mass.
        // Jupiter with 100 times its normal mass.
        // Jupiter with 1000 times its normal mass. (This one is fun!)

        integrationParameters[0] = 0.01;  // time step in years
        integrationParameters[1] = 15.2;  // total time in years

        integratorChoice = 3;  // Verlet

        switch (choice) {
            case 1: {
                // Setup Earth-Sun system
                CelestialObject sun(1.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, "Sun");
                CelestialObject earth(3.0027e-6, {1.0, 0.0, 0.0}, {0.0, 2 * M_PI, 0.0}, "Earth");
                system.addObject(sun);
                system.addObject(earth);

                break;
            }
            case 2: {
                // Setup Jupiter-Sun system
                CelestialObject sun(1.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, "Sun");
                CelestialObject jupiter(9.54588e-4, {-5.2, 0.0, 0.0}, {0.0, -0.876897 * M_PI, 0.0},
                                        "Jupiter");
                system.addObject(sun);
                system.addObject(jupiter);
                break;
            }
            case 3: {
                // Setup binary Sun system
                CelestialObject sun1(0.5, {0, 0.0, 0.0}, {0.0, 0, 0.0}, "Sun 1");
                CelestialObject sun2(0.5, {-5.2, 0.0, 0.0}, {0.0, -0.876897 * M_PI, 0.0}, "Sun 2");
                system.addObject(sun1);
                system.addObject(sun2);
                break;
            }
            case 4: {
                // Setup Sun-Earth-Moon system
                CelestialObject sun(1.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, "Sun");
                CelestialObject earth(3.0027e-6, {1.0, 0.0, 0.0}, {0.0, 2 * M_PI, 0.0}, "Earth");
                CelestialObject moon(3.694e-8, {1.00257, 0.0, 0.0},
                                     {0.0, 2 * M_PI + 0.0748 * M_PI, 0.0}, "Moon");
                system.addObject(sun);
                system.addObject(earth);
                system.addObject(moon);
                break;
            }
            case 5: {
                // Setup Sun-Earth-Jupiter system
                std::cout << "Enter mass multiplier for Jupiter (e.g., 1, 10, 100, 1000): ";
                double massMultiplier;
                std::cin >> massMultiplier;
                CelestialObject sun(1.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, "Sun");
                CelestialObject earth(3.0027e-6, {1.0, 0.0, 0.0}, {0.0, 2 * M_PI, 0.0}, "Earth");
                CelestialObject jupiter(9.54588e-4 * massMultiplier, {-5.2, 0.0, 0.0},
                                        {0.0, -0.876897 * M_PI, 0.0}, "Jupiter");
                system.addObject(sun);
                system.addObject(earth);
                system.addObject(jupiter);
                break;
            }
            default:
                std::cout << "Invalid choice. Exiting verification simulation." << std::endl;
                return;
        }
    }

    void SolarSystemSimulation(CelestialSystem& system,
                               std::array<double, 2>& integrationParameters,
                               int& integratorChoice) {
        // Implementation of solar system simulation
        // Masses in solar masses, positions in AU, velocities in AU/year
        std::cout
            << "Do you want to do a normal solar system simulation (1) or set mass multiplier for "
               "Jupiter (2)? (1/2): ";
        int choice;
        double massMultiplier = 1.0;
        std::cin >> choice;
        if (choice == 2) {
            std::cout << "Enter mass multiplier for Jupiter (e.g., 1, 10, 100, 1000): ";
            std::cin >> massMultiplier;
        }

        CelestialObject sun(1.0, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, "Sun");
        CelestialObject mercury(1.66e-7, {0.387, 0.0, 0.0}, {0.0, 10.05, 0.0}, "Mercury");
        CelestialObject venus(2.45e-6, {0.723, 0.0, 0.0}, {0.0, 7.38, 0.0}, "Venus");
        CelestialObject earth(3.003e-6, {1.0, 0.0, 0.0}, {0.0, 6.283, 0.0}, "Earth");
        CelestialObject mars(3.23e-7, {1.524, 0.0, 0.0}, {0.0, 5.08, 0.0}, "Mars");
        CelestialObject jupiter(9.55e-4 * massMultiplier, {5.203, 0.0, 0.0}, {0.0, 2.76, 0.0},
                                "Jupiter");
        CelestialObject saturn(2.86e-4, {9.537, 0.0, 0.0}, {0.0, 2.04, 0.0}, "Saturn");
        CelestialObject uranus(4.37e-5, {19.191, 0.0, 0.0}, {0.0, 1.43, 0.0}, "Uranus");
        CelestialObject neptune(5.15e-5, {30.07, 0.0, 0.0}, {0.0, 1.14, 0.0}, "Neptune");
        CelestialObject pluto(6.55e-9, {39.48, 0.0, 0.0}, {0.0, 0.99, 0.0}, "Pluto");

        system.addObject(sun);
        system.addObject(mercury);
        system.addObject(venus);
        system.addObject(earth);
        system.addObject(mars);
        system.addObject(jupiter);
        system.addObject(saturn);
        system.addObject(uranus);
        system.addObject(neptune);
        system.addObject(pluto);
        integrationParameters[0] = 0.01;  // time step in years
        integrationParameters[1] = 30.0;  // total time in years
        integratorChoice = 3;             // Verlet
    }

    void CustomSimulation(CelestialSystem& system, std::array<double, 2>& integrationParameters,
                          int& integratorChoice) {
        while (true) {
            CelestialObject obj;

            system.addObject(obj);
            cout << "Add another object? (y/n): ";
            char cont;
            cin >> cont;
            if (cont != 'y' && cont != 'Y') {
                break;
            }
        }

        GetIntegrationParameters(integrationParameters, integratorChoice);
    }

    void GetIntegrationParameters(std::array<double, 2>& integrationParameters,
                                  int& integratorChoice) {
        cout << "Enter time step (in years): ";
        cin >> integrationParameters[0];
        cout << "Enter total simulation time (in years): ";
        cin >> integrationParameters[1];
        cout << "Choose integrator - RK4(1), Euler-Cromer(2), Verlet(3): ";
        cin >> integratorChoice;
    }

    void outputResults(const std::vector<std::vector<double>>& trajectory,
                       const CelestialSystem& system, const std::string& dataInfo) {
        // Implementation for outputting results
        int i = 0;
        std::string baseFilename =
            "/Users/nelsbuhrley/CPP_Workspace/Project 3: Celestial "
            "Dynamics/Output/celestial_output";
        std::string filename;
        std::string intermediateFilename = baseFilename + "_" + std::to_string(i);

        while (std::ifstream(intermediateFilename + ".csv").good()) {
            intermediateFilename = baseFilename + "_" + std::to_string(i);
            i++;
        }
        filename = intermediateFilename + ".csv";

        std::ofstream outFile(filename);
        if (outFile.is_open()) {
            // Write optional metadata header (typically comments prefixed with #)
            outFile << dataInfo << "\n";

            // Write CSV header row
            outFile << "Time";
            std::vector<std::string> objectNames = system.getObjectNames();
            for (const auto& name : objectNames) {
                outFile << "," << name << "_x"
                        << "," << name << "_y"
                        << "," << name << "_z";
            }
            outFile << "\n";

            // Write trajectory data - one row per time step
            for (const auto& state : trajectory) {
                for (size_t j = 0; j < state.size(); ++j) {
                    outFile << state[j];
                    if (j < state.size() - 1) {
                        outFile << ",";
                    }
                }
                outFile << "\n";
            }
            outFile.close();
            std::cout << "Results written to " << filename << std::endl;
        } else {
            std::cerr << "Error: Unable to open output file." << std::endl;
        }

        // Call the Python plotting program to generate visualizations
        // The plotting script will create 2D and 3D orbital path plots
        std::string plotCommand = "python3 plotting.py \"" + filename + "\"";

        std::cout << "Executing plot command: " << plotCommand << std::endl;
        int plotResult = std::system(plotCommand.c_str());  // Execute system command
        if (plotResult != 0) {
            std::cerr << "Error: plotting command failed with code " << plotResult << std::endl;
        }
    }
};