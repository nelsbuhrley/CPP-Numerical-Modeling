#include "oscillator.h"

// TODO: Define your oscillator model and integration routines here.

// A mass of 1.00 kg.
// A pendulum length of 9.8 meters.
// A damping coefficient of 0.5.
// An initial angle of 0.20 radians.
// An initial angular velocity of zero.
// Model 180 seconds of motion.
// Use a time step small enough that your results are stable.
// A driving force of 1.20 N.
// A driving frequency of 2/3 rad/s.

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

void oscillator::computeDerivatives(const std::vector<double>& state,
                                    std::vector<double>& derivatives, double timeStep) {
    double time = state[0];
    double angle = state[1];
    double angularVelocity = state[2];
    derivatives[0] = 1.0;

    // Derivative of angle is angular velocity
    derivatives[1] = angularVelocity;
    // Derivative of angular velocity (equation of motion)
    double gravityTerm = -(9.8 / length) * sin(angle);
    double dampingTerm = -(dampingCoefficient / mass) * angularVelocity;
    double drivingTerm = (drivingForce / mass) * sin(drivingFrequency * time);

    derivatives[2] = gravityTerm + dampingTerm + drivingTerm;
};

std::vector<double> oscillator::getState() const {
    return {time, angle, angularVelocity};
}

void oscillator::printParameters() const {
    std::cout << "Mass: " << mass << " kg\n"
              << "Length: " << length << " m\n"
              << "Damping Coefficient: " << dampingCoefficient << "\n"
              << "Initial Angle: " << angle << " rad\n"
              << "Initial Angular Velocity: " << angularVelocity << " rad/s\n"
              << "Driving Force: " << drivingForce << " N\n"
              << "Driving Frequency: " << drivingFrequency << " rad/s\n";
}

