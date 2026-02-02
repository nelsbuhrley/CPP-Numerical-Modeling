#include <atomic>
#include <bitset>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

#include "run.h"

// Initialiise a Run object and start the simulation
int main() {
    Run app;  // Create the application controller
    app.runSimulation();  // Launch the interactive simulation interface
    return 0;
}