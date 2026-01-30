
#pragma once
#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "objects.h"
/**
 * @file objects.h
 * @brief Defines CelestialObject and CelestialSystem classes for celestial dynamics simulations
 *
 * @author Nels Buhrley
 * @date June 2024
 * This header defines two main classes:
 * - CelestialObject: Represents a celestial body with mass, position, and velocity
 * - CelestialSystem: Manages a collection of CelestialObject instances and computes gravitational interactions
 * All units are in Astronomical Units (AU), Solar Masses, and Years for consistency.
 *
 * Usage:
 * - Create CelestialObject instances for each body in the simulation
 * - Add them to a CelestialSystem instance
 * - Use CelestialSystem methods to compute forces and evolve the system over time
 * - Gravitational constant G is defined in appropriate units for these calculations
 *
 * 
 */


// Gravitational constant in AU^3 / (Solar Mass * Year^2)
// G = 4 * pi^2 in these units (since 1 AU, 1 year, 1 Solar Mass gives this value)
constexpr double G = 4 * M_PI * M_PI;

class CelestialObject {
    /**
     * CelestialObject represents a body in space with mass, position, and velocity.
     * It provides methods to compute gravitational interactions with other objects.
     *
     * Attributes:
     * - mass: Mass of the object (Solar masses)
     * - position: 3D position vector (AU)
     * - velocity: 3D velocity vector (AU/year)
     *
     * Methods:
     * - computeGravitationalForce: Calculates gravitational force exerted by another object.
     *
     */
   public:
    double mass;                     ///< Mass of the celestial object (Solar masses)
    std::array<double, 3> position;  ///< Position vector in 3D space (AU)
    std::array<double, 3> velocity;  ///< Velocity vector in 3D space (AU/year)
    std::string name;                ///< Name identifier

    CelestialObject(double mass_, const std::array<double, 3>& position_,
                    const std::array<double, 3>& velocity_, const std::string& name_ = "")
        : mass(mass_), position(position_), velocity(velocity_), name(name_) {}

    CelestialObject() {
        std::cout << "Creating a new Celestial Object." << std::endl;
        std::cout << "Please Enter Name" << std::endl;
        std::cin >> name;
        std::cout << "Please Enter Mass (Solar masses)" << std::endl;
        std::cin >> mass;
        std::cout << "Please Enter Position as x y z (AU)" << std::endl;
        std::cin >> position[0] >> position[1] >> position[2];
        std::cout << "Please Enter Velocity as vx vy vz (AU/year)" << std::endl;
        std::cin >> velocity[0] >> velocity[1] >> velocity[2];
    }

    std::string getInfo() const {
        return "#Object: " + name + "\n"                                  //
               + "#   Mass: " + std::to_string(mass) + " Solar masses\n"  //
               + "#   Position: (" + std::to_string(position[0]) + ", " +
               std::to_string(position[1]) + ", " + std::to_string(position[2]) + ") AU\n"  //
               + "#   Velocity: (" + std::to_string(velocity[0]) + ", " +
               std::to_string(velocity[1]) + ", " + std::to_string(velocity[2]) + ") AU/year\n";
    }

    /**
     * @brief Computes the gravitational force exerted by another object on this one
     *
     * Uses Newton's law of gravitation: F = G * m1 * m2 / r^2
     * Returns the force vector pointing from this object toward the other.
     *
     * @param other The other celestial object exerting the force
     * @return Force vector (in Solar masses * AU / year^2)
     */
    std::array<double, 3> computeGravitationalForce(const CelestialObject& other) const {
        // Compute displacement vector from this object to the other
        std::array<double, 3> r = {other.position[0] - position[0], other.position[1] - position[1],
                                   other.position[2] - position[2]};

        // Compute distance (magnitude of displacement)
        double distance = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

        // Avoid division by zero for coincident objects
        if (distance < 1e-10) {
            return {0.0, 0.0, 0.0};
        }

        // Compute gravitational force magnitude: F = G * m1 * m2 / r^2
        double forceMagnitude = G * mass * other.mass / (distance * distance);

        // Return force vector (normalized direction * magnitude)
        return {forceMagnitude * r[0] / distance, forceMagnitude * r[1] / distance,
                forceMagnitude * r[2] / distance};
    }
};

class CelestialSystem {
    /**
     * CelestialSystem manages a collection of CelestialObject instances.
     * It provides methods to add objects and compute pairwise gravitational forces.
     *
     * Attributes:
     * - objects: Vector of CelestialObject instances in the system
     *
     * Methods:
     * - addObject: Adds a new CelestialObject to the system
     * - computeAllForces: Computes gravitational forces between all pairs of objects
     *
     */
   public:
    std::vector<CelestialObject> objects;  ///< Collection of celestial objects

    /**
     * @brief Adds a celestial object to the system and then places the origin at the center of mass
     * of the system and adjusts the velocities of the system to ensure total momentum is zero.
     * while maintaining the relative velocities.
     * @param obj The object to add
     */
    void addObject(const CelestialObject& obj) {
        objects.push_back(obj);
        // Recalculate center of mass and adjust positions and velocities
        double totalMass = 0.0;
        std::array<double, 3> comPosition = {0.0, 0.0, 0.0};
        std::array<double, 3> totalMomentum = {0.0, 0.0, 0.0};

        for (const auto& o : objects) {
            totalMass += o.mass;
            comPosition[0] += o.position[0] * o.mass;
            comPosition[1] += o.position[1] * o.mass;
            comPosition[2] += o.position[2] * o.mass;

            totalMomentum[0] += o.velocity[0] * o.mass;
            totalMomentum[1] += o.velocity[1] * o.mass;
            totalMomentum[2] += o.velocity[2] * o.mass;
        }
        comPosition[0] /= totalMass;
        comPosition[1] /= totalMass;
        comPosition[2] /= totalMass;

        std::array<double, 3> comVelocity = {totalMomentum[0] / totalMass,
                                             totalMomentum[1] / totalMass,
                                             totalMomentum[2] / totalMass};

        for (auto& o : objects) {
            o.position[0] -= comPosition[0];
            o.position[1] -= comPosition[1];
            o.position[2] -= comPosition[2];

            o.velocity[0] -= comVelocity[0];
            o.velocity[1] -= comVelocity[1];
            o.velocity[2] -= comVelocity[2];
        }
    }

    std::vector<std::string> getObjectNames() const {
        std::vector<std::string> names;
        for (const auto& obj : objects) {
            names.push_back(obj.name);
        }
        return names;
    }


    /**
     * @brief Computes all gravitational accelerations for all objects
     *
     * Returns a flat vector containing accelerations for each object:
     * [ax1, ay1, az1, ax2, ay2, az2, ...]
     *
     * @return Vector of acceleration components
     */
    std::vector<double> computeAllForces() const {
        std::vector<double> accelerations(objects.size() * 3, 0.0);

        // Compute pairwise forces and convert to accelerations
        for (size_t i = 0; i < objects.size(); ++i) {
            for (size_t j = 0; j < objects.size(); ++j) {
                if (i != j) {
                    // Get force on object i from object j
                    std::array<double, 3> force = objects[i].computeGravitationalForce(objects[j]);

                    // Convert force to acceleration: a = F / m
                    accelerations[i * 3 + 0] += force[0] / objects[i].mass;
                    accelerations[i * 3 + 1] += force[1] / objects[i].mass;
                    accelerations[i * 3 + 2] += force[2] / objects[i].mass;
                }
            }
        }

        return accelerations;
    }

    /**
     * @brief Creates a state vector from the current system state
     *
     * Format: [time, x1, vx1, y1, vy1, z1, vz1, x2, vx2, ...]
     * This format is compatible with the integrator functions.
     *
     * @param time Current simulation time
     * @return State vector for integration
     */
    std::vector<double> getStateVector(double time = 0.0) const {
        std::vector<double> state;
        state.push_back(time);

        for (const auto& obj : objects) {
            state.push_back(obj.position[0]);
            state.push_back(obj.velocity[0]);
            state.push_back(obj.position[1]);
            state.push_back(obj.velocity[1]);
            state.push_back(obj.position[2]);
            state.push_back(obj.velocity[2]);
        }

        return state;
    }

    /**
     * @brief Updates the system from a state vector
     *
     * @param state State vector in format [time, x1, vx1, y1, vy1, z1, vz1, x2, ...]
     */
    void setFromStateVector(const std::vector<double>& state) {
        size_t idx = 1;  // Skip time at index 0
        for (auto& obj : objects) {
            obj.position[0] = state[idx++];
            obj.velocity[0] = state[idx++];
            obj.position[1] = state[idx++];
            obj.velocity[1] = state[idx++];
            obj.position[2] = state[idx++];
            obj.velocity[2] = state[idx++];
        }
    }

    /**
     * @brief Returns a derivative function compatible with the integrators
     *
     * The returned function computes derivatives for the state vector:
     * - Position derivatives = velocities
     * - Velocity derivatives = accelerations (from gravity)
     *
     * @return Lambda function for use with rk4, euler_chromer, or verlet
     */
    std::function<void(const std::vector<double>&, std::vector<double>&, double)>
    getDerivativeFunction() {
        return [this](const std::vector<double>& state, std::vector<double>& derivs, double /*t*/) {
            // Update object positions/velocities from state vector temporarily
            CelestialSystem tempSystem = *this;
            tempSystem.setFromStateVector(state);

            // Compute accelerations
            std::vector<double> accelerations = tempSystem.computeAllForces();

            // Fill in derivatives: d(pos)/dt = vel, d(vel)/dt = accel
            derivs[0] = 1.0;  // d(time)/dt = 1

            size_t accelIdx = 0;
            for (size_t i = 1; i < state.size(); i += 2) {
                // Position derivative = velocity (which is at i+1)
                derivs[i] = state[i + 1];
                // Velocity derivative = acceleration
                derivs[i + 1] = accelerations[accelIdx++];
            }
        };
    }

    // ==================== Verlet-compatible methods ====================

    /**
     * @brief Creates a position-only state vector for Verlet integration
     *
     * Format: [time, x1, y1, z1, x2, y2, z2, ..., xN, yN, zN]
     * This format is specifically designed for the Verlet integrator which
     * only tracks positions (velocities are derived from position history).
     *
     * @param time Current simulation time
     * @return Position-only state vector
     */
    std::vector<double> getVerletStateVector(double time = 0.0) const {
        std::vector<double> state;
        state.push_back(time);

        for (const auto& obj : objects) {
            state.push_back(obj.position[0]);  // x
            state.push_back(obj.position[1]);  // y
            state.push_back(obj.position[2]);  // z
        }

        return state;
    }

    /**
     * @brief Gets velocities as a flat vector for Verlet initialization
     *
     * Format: [vx1, vy1, vz1, vx2, vy2, vz2, ..., vxN, vyN, vzN]
     * Used with initializeVerletPreviousState() to bootstrap Verlet integration.
     *
     * @return Velocity vector (no time component)
     */
    std::vector<double> getVelocityVector() const {
        std::vector<double> velocities;

        for (const auto& obj : objects) {
            velocities.push_back(obj.velocity[0]);  // vx
            velocities.push_back(obj.velocity[1]);  // vy
            velocities.push_back(obj.velocity[2]);  // vz
        }

        return velocities;
    }

    /**
     * @brief Updates the system positions from a Verlet state vector
     *
     * @param state Position-only state vector [time, x1, y1, z1, x2, y2, z2, ...]
     */
    void setFromVerletStateVector(const std::vector<double>& state) {
        size_t idx = 1;  // Skip time at index 0
        for (auto& obj : objects) {
            obj.position[0] = state[idx++];  // x
            obj.position[1] = state[idx++];  // y
            obj.position[2] = state[idx++];  // z
        }
    }

    /**
     * @brief Updates velocities from two consecutive Verlet states
     *
     * Computes velocities using centered difference: v = (x_{n+1} - x_{n-1}) / (2h)
     *
     * @param currentState Current positions
     * @param previousState Previous positions
     * @param timeStep Integration step size
     */
    void computeVelocitiesFromVerlet(const std::vector<double>& currentState,
                                     const std::vector<double>& previousState, double timeStep) {
        size_t idx = 1;  // Skip time at index 0
        for (auto& obj : objects) {
            obj.velocity[0] = (currentState[idx] - previousState[idx]) / (2.0 * timeStep);
            idx++;
            obj.velocity[1] = (currentState[idx] - previousState[idx]) / (2.0 * timeStep);
            idx++;
            obj.velocity[2] = (currentState[idx] - previousState[idx]) / (2.0 * timeStep);
            idx++;
        }
    }

    /**
     * @brief Returns an acceleration function compatible with Verlet integrator
     *
     * The returned function computes gravitational accelerations from positions only.
     * Format: accelerations[0] = 0 (time), accelerations[1..3] = ax1,ay1,az1, etc.
     *
     * @return Lambda function for use with verlet()
     */
    std::function<void(const std::vector<double>&, std::vector<double>&)>
    getVerletAccelerationFunction() {
        return [this](const std::vector<double>& positions, std::vector<double>& accelerations) {
            // Update object positions from state vector temporarily
            CelestialSystem tempSystem = *this;
            tempSystem.setFromVerletStateVector(positions);

            // Compute accelerations
            std::vector<double> accels = tempSystem.computeAllForces();

            // Fill in accelerations array
            // accelerations[0] = 0 (no acceleration for time)
            accelerations[0] = 0.0;

            // Copy accelerations for each object's x, y, z components
            for (size_t i = 0; i < accels.size(); ++i) {
                accelerations[i + 1] = accels[i];
            }
        };
    }
};
