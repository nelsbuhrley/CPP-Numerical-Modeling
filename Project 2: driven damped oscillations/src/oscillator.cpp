/**
 * @file oscillator.cpp
 * @brief Implementation of oscillator classes for driven damped pendulum simulation
 *
 * Implements the physics model for a driven damped pendulum system, including
 * equations of motion, state management, and parameter display.
 */

#include "oscillator.h"

// TODO: Define your oscillator model and integration routines here.

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
 * @brief Constructs an oscillator with specified physical parameters
 *
 * Initializes all member variables and sets initial simulation time to 0.
 * The state vector is not populated here; use getState() to retrieve initial conditions.
 *
 * @param mass_ Mass of the pendulum bob (kg)
 * @param length_ Length of the pendulum (m)
 * @param dampingCoefficient_ Damping coefficient (affects energy dissipation)
 * @param initialAngle_ Initial angular displacement from vertical (radians)
 * @param initialAngularVelocity_ Initial angular velocity (rad/s)
 * @param drivingForce_ Amplitude of periodic driving force (N)
 * @param drivingFrequency_ Driving frequency (rad/s)
 */
oscillator::oscillator(double mass_, double length_, double dampingCoefficient_,
                       double initialAngle_, double initialAngularVelocity_, double drivingForce_,
                       double drivingFrequency_)
    : mass(mass_),
      length(length_),
      dampingCoefficient(dampingCoefficient_),
      angle(initialAngle_),
      angularVelocity(initialAngularVelocity_),
      drivingForce(drivingForce_),
      drivingFrequency(drivingFrequency_),
      time(0.0) {}

/**
 * @brief Computes time derivatives for the driven damped pendulum system
 *
 * Implements the equations of motion for a driven damped pendulum:
 *
 * Governing equation:
 *   θ'' + (q/m)θ' + (g/L)sin(θ) = (F_D/m)sin(ω_D·t)
 *
 * Where:
 *   θ   = angle from vertical (radians)
 *   θ'  = angular velocity (rad/s)
 *   θ'' = angular acceleration (rad/s²)
 *   q   = damping coefficient
 *   m   = mass (kg)
 *   g   = gravitational acceleration (9.8 m/s²)
 *   L   = pendulum length (m)
 *   F_D = driving force amplitude (N)
 *   ω_D = driving frequency (rad/s)
 *
 * State vector format: [time, angle, angular_velocity]
 * Derivative vector format: [dt/dt=1, dθ/dt=ω, dω/dt=α]
 *
 * @param state Current state vector [t, θ, ω]
 * @param derivatives Output array for derivatives [1, ω, α]
 * @param timeStep Time increment (not used in this implementation, kept for interface
 * compatibility)
 */
void oscillator::computeDerivatives(const std::vector<double>& state,
                                    std::vector<double>& derivatives, double timeStep) {
    // Extract state variables for clarity
    double time = state[0];
    double angle = state[1];
    double angularVelocity = state[2];

    // Time always advances at rate 1
    derivatives[0] = 1.0;

    // Derivative of angle is angular velocity (dθ/dt = ω)
    derivatives[1] = angularVelocity;

    // Derivative of angular velocity (equation of motion for angular acceleration)
    // α = -(g/L)sin(θ) - (q/m)ω + (F_D/m)sin(ω_D·t)

    double gravityTerm = -(9.8 / length) * sin(angle);  // Restoring torque from gravity
    double dampingTerm = -(dampingCoefficient / mass) * angularVelocity;  // Energy dissipation
    double drivingTerm = (drivingForce / mass) * sin(drivingFrequency * time);  // External forcing

    // Sum all contributions to get total angular acceleration
    derivatives[2] = gravityTerm + dampingTerm + drivingTerm;
};

/**
 * @brief Returns the current state as a vector
 *
 * Packages the current simulation state into a vector suitable for
 * use with numerical integrators. Format: [time, angle, angular_velocity]
 *
 * @return State vector containing [t, θ, ω]
 */
std::vector<double> oscillator::getState() const {
    return {time, angle, angularVelocity};
}

/**
 * @brief Prints all oscillator parameters to standard output
 *
 * Displays a formatted summary of all physical parameters and initial conditions.
 * Useful for verification and documentation of simulation runs.
 */
void oscillator::printParameters() const {
    std::cout << "Mass: " << mass << " kg\n"
              << "Length: " << length << " m\n"
              << "Damping Coefficient: " << dampingCoefficient << "\n"
              << "Initial Angle: " << angle << " rad\n"
              << "Initial Angular Velocity: " << angularVelocity << " rad/s\n"
              << "Driving Force: " << drivingForce << " N\n"
              << "Driving Frequency: " << drivingFrequency << " rad/s\n";
}
