/**
 * @file main.cpp
 * @brief Entry point for the driven damped oscillator simulation program
 *
 * This program simulates a driven damped pendulum/oscillator system using numerical
 * integration methods (RK4 or Euler-Cromer). The simulation can be run in validation
 * mode with predefined parameters or in custom mode with user-specified parameters.
 */

#include <fstream>
#include <iostream>
#include <vector>

#include "oscillator.h"
#include "processing.h"

/**
 * @brief Main entry point for the oscillator simulation
 *
 * Creates a logic controller object and runs the interactive simulation program.
 * The logic class handles user interaction, parameter selection, numerical integration,
 * data output, and plotting.
 *
 * @return 0 on successful program execution
 */
int main() {
    logic app;  // Create the application controller
    app.run();  // Launch the interactive simulation interface
    return 0;
}
