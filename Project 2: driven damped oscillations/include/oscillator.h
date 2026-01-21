#pragma once

#include <cmath>
#include <iostream>
#include <vector>  // Include the vector header

#include "processing.h"

// TODO: Declare your driven damped oscillator types and interfaces here.
// Suggestion: a struct for parameters (mass, damping, drive amplitude/frequency)
// and a function/class to step the ODE (e.g., using RK4 or another integrator).

// A mass of 1.00 kg.
// A pendulum length of 9.8 meters.
// A damping coefficient of 0.5.
// An initial angle of 0.20 radians.
// An initial angular velocity of zero.
// Model 180 seconds of motion.
// Use a time step small enough that your results are stable.
// A driving force of 1.20 N.
// A driving frequency of 2/3 rad/s.

class oscillator {
   public:
    double mass;
    double length;
    double dampingCoefficient;
    double angle;
    double angularVelocity;
    double drivingForce;
    double drivingFrequency;
    double time;
    std::vector<double> state;
    std::vector<double> derivatives;

    oscillator(double mass_, double length_, double dampingCoefficient_, double initialAngle_,
               double initialAngularVelocity_, double drivingForce_, double drivingFrequency_);

    void computeDerivatives(const std::vector<double>& state, std::vector<double>& derivatives,
                            double time);

    std::vector<double> getState() const;

    void printParameters() const;

    // Returns a lambda function that wraps computeDerivatives for use with rk4Simulation
    auto getDerivativeFunction() {
        return [this](const std::vector<double>& state, std::vector<double>& derivatives,
                      double time) { computeDerivatives(state, derivatives, time); };
    }

    // Returns a stop condition lambda that continues while time < maxTime
    auto getStopCondition(double maxTime = 180.0) {
        return [maxTime](const std::vector<double>& state) { return state[0] < maxTime; };
    }
};

// A mass of 1.00 kg.
// A pendulum length of 9.8 meters.
// A damping coefficient of 0.5.
// An initial angle of 0.20 radians.
// An initial angular velocity of zero.
// Model 180 seconds of motion.
// Use a time step small enough that your results are stable.
// A driving force of 1.20 N.
// A driving frequency of 2/3 rad/s.

//  oscillator(double mass_, double length_, double dampingCoefficient_, double initialAngle_,
//                double initialAngularVelocity_, double drivingForce_, double drivingFrequency_);

class testOscillator : public oscillator {
   public:
    testOscillator() : oscillator(1.0, 9.8, 0.5, 0.2, 0.0, 1.2, 2.0 / 3.0) {}
};
