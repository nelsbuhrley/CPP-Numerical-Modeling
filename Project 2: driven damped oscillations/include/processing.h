#pragma once

#include <functional>
#include <vector>

// TODO: Declare your data processing functions here.

namespace integrator {
typedef std::vector<double> state_type;

// RK4 simulation function
std::vector<state_type> rk4(state_type& state,
                            std::function<void(const state_type&, state_type&, double)> derivatives,
                            std::function<bool(const state_type&)> stopCondition, double timeStep);

std::vector<state_type> euler_chromer(
    state_type& state, std::function<void(const state_type&, state_type&, double)> derivatives,
    std::function<bool(const state_type&)> stopCondition, double timeStep);
}  // namespace integrator

class logic {
   public:
    void run();
    void outputResults(const std::vector<std::vector<double>>& path, const std::string& filename,
                       double maxPlotTime, double driverFrequency, std::string dataInfo = "");
    void runValidationTest();
    void runCustomSimulation();
};