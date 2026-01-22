/**
 * @file oscillator.h
 * @brief Defines oscillator classes for simulating driven damped pendulum systems
 *
 * This header provides the oscillator class which models a driven damped pendulum
 * governed by the equation: θ'' + (q/m)θ' + (g/L)sin(θ) = (F_D/m)cos(ω_D·t)
 * Also includes testOscillator class with predefined validation parameters.
 */

#pragma once

#include <cmath>
#include <iostream>
#include <vector>  // Include the vector header

#include "processing.h"

// TODO: Declare your driven damped oscillator types and interfaces here.
// Suggestion: a struct for parameters (mass, damping, drive amplitude/frequency)
// and a function/class to step the ODE (e.g., using RK4 or another integrator).

/**
 * Validation Test Parameters (from project requirements):
 * - A mass of 1.00 kg.
 * - A pendulum length of 9.8 meters.
 * - A damping coefficient of 0.5.
 * - An initial angle of 0.20 radians.
 * - An initial angular velocity of zero.
 * - Model 180 seconds of motion.
 * - Use a time step small enough that your results are stable.
 * - A driving force of 1.20 N.
 * - A driving frequency of 2/3 rad/s.
 */

/**
 * @class oscillator
 * @brief Models a driven damped pendulum system
 *
 * This class encapsulates all physical parameters and provides methods for
 * computing derivatives, managing state, and interfacing with numerical integrators.
 */
class oscillator {
   public:
    // Physical parameters
    double mass;                ///< Mass of the pendulum bob (kg)
    double length;              ///< Length of the pendulum (m)
    double dampingCoefficient;  ///< Damping coefficient (dimensionless or kg/s)
    double angle;               ///< Current angle from vertical (radians)
    double angularVelocity;     ///< Current angular velocity (rad/s)
    double drivingForce;        ///< Amplitude of driving force (N)
    double drivingFrequency;    ///< Driving frequency (rad/s)
    double time;                ///< Current simulation time (s)

    // State vectors for integration
    std::vector<double> state;        ///< State vector [time, angle, angular_velocity]
    std::vector<double> derivatives;  ///< Derivative vector [1, dθ/dt, dω/dt]

    /**
     * @brief Constructs an oscillator with specified parameters
     *
     * @param mass_ Mass of the pendulum bob (kg)
     * @param length_ Length of the pendulum (m)
     * @param dampingCoefficient_ Damping coefficient
     * @param initialAngle_ Initial angle from vertical (radians)
     * @param initialAngularVelocity_ Initial angular velocity (rad/s)
     * @param drivingForce_ Amplitude of driving force (N)
     * @param drivingFrequency_ Driving frequency (rad/s)
     */
    oscillator(double mass_, double length_, double dampingCoefficient_, double initialAngle_,
               double initialAngularVelocity_, double drivingForce_, double drivingFrequency_);

    /**
     * @brief Computes time derivatives for the driven damped pendulum
     *
     * Implements the equations of motion:
     * - dθ/dt = ω (angular velocity)
     * - dω/dt = -(g/L)sin(θ) - (q/m)ω + (F_D/m)sin(ω_D·t)
     *
     * @param state Current state vector [time, angle, angular_velocity]
     * @param derivatives Output derivative vector [1, dθ/dt, dω/dt]
     * @param time Current simulation time (used for driving term)
     */
    void computeDerivatives(const std::vector<double>& state, std::vector<double>& derivatives,
                            double time);

    /**
     * @brief Returns the current state vector
     * @return State vector containing [time, angle, angular_velocity]
     */
    std::vector<double> getState() const;

    /**
     * @brief Prints all oscillator parameters to console
     */
    void printParameters() const;

    /**
     * @brief Returns a lambda function that wraps computeDerivatives for use with rk4Simulation
     *
     * This provides a convenient interface for numerical integrators that expect a
     * function object matching the signature: void(const vector&, vector&, double)
     *
     * @return Lambda function wrapping this oscillator's computeDerivatives method
     */
    auto getDerivativeFunction() {
        return [this](const std::vector<double>& state, std::vector<double>& derivatives,
                      double time) { computeDerivatives(state, derivatives, time); };
    }

    /**
     * @brief Returns a stop condition lambda that continues while time < maxTime
     *
     * Used by integrators to determine when to halt the simulation. The condition
     * returns true to continue and false to stop.
     *
     * @param maxTime Maximum simulation time (default: 180.0 seconds)
     * @return Lambda function that checks if state[0] (time) < maxTime
     */
    auto getStopCondition(double maxTime = 180.0) {
        return [maxTime](const std::vector<double>& state) { return state[0] < maxTime; };
    }
};

/**
 * Validation Test Parameters (repeated for reference):
 * - A mass of 1.00 kg.
 * - A pendulum length of 9.8 meters.
 * - A damping coefficient of 0.5.
 * - An initial angle of 0.20 radians.
 * - An initial angular velocity of zero.
 * - Model 180 seconds of motion.
 * - Use a time step small enough that your results are stable.
 * - A driving force of 1.20 N.
 * - A driving frequency of 2/3 rad/s.
 */

//  oscillator(double mass_, double length_, double dampingCoefficient_, double initialAngle_,
//                double initialAngularVelocity_, double drivingForce_, double drivingFrequency_);

/**
 * @class testOscillator
 * @brief Predefined oscillator configuration for validation testing
 *
 * Inherits from oscillator and initializes with the standard validation parameters
 * specified in the project requirements. Useful for benchmarking and verification.
 */
class testOscillator : public oscillator {
   public:
    /**
     * @brief Constructs a test oscillator with standard validation parameters
     *
     * Parameters: mass=1.0kg, length=9.8m, damping=0.5, angle=0.2rad,
     *             velocity=0.0rad/s, force=1.2N, frequency=2/3 rad/s
     */
    testOscillator() : oscillator(1.0, 9.8, 0.5, 0.2, 0.0, 1.2, 2.0 / 3.0) {}
};
