/**
 * @file processing.h
 * @brief Declares numerical integration methods and simulation control logic
 *
 * This header provides:
 * - integrator namespace: Contains RK4, Euler-Cromer, and Verlet integration methods
 * - All implementations are header-only for template support
 * Author: Nels Buhrley
 *
 *  Date: June 2024
 *
 * Usage:
 * - Include this header to access integrator methods
 * - Call integrator::rk4, integrator::euler_chromer, or integrator::verlet with appropriate
 * parameters
 *
 * Limitations:
 * - Fixed time step methods; adaptive stepping not implemented
 * - Not suitable for stiff equations without modification
 * Limitations and error estimates for each method are provided in the respective documentation.
 */

#pragma once

#include <array>
#include <functional>
#include <vector>

/**
 * @namespace integrator
 * @brief Contains numerical integration algorithms for ODE systems
 *
 * Provides three integration methods optimized for different scenarios:
 * - RK4: High-accuracy 4th-order Runge-Kutta method
 * - Euler-Cromer: Semi-implicit method with good energy conservation for oscillators
 * - Verlet: Symplectic integrator for conservative systems like celestial mechanics
 */
namespace integrator {

/**
 * @brief 4th-order Runge-Kutta (RK4) integration method
 *
 * A highly accurate explicit method for solving ordinary differential equations.
 * Uses four intermediate evaluations per step to achieve O(h^5) local truncation error.
 *
 * Algorithm per step:
 * - k1 = f(t_n, y_n)
 * - k2 = f(t_n + h/2, y_n + h*k1/2)
 * - k3 = f(t_n + h/2, y_n + h*k2/2)
 * - k4 = f(t_n + h, y_n + h*k3)
 * - y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
 *
 * @param state Initial state vector (modified in-place during integration)
 * @param derivatives Function that computes dy/dt given (y, dy/dt, t)
 * @param stopCondition Function returning true to continue, false to stop
 * @param timeStep Integration step size (smaller = more accurate but slower)
 * @return Trajectory of state vectors at each time step
 */
template <typename StateType = std::vector<double>>
std::vector<StateType> rk4(StateType& state,
                           std::function<void(const StateType&, StateType&, double)> derivatives,
                           std::function<bool(const StateType&)> stopCondition, double timeStep) {
    //
    std::vector<StateType> trajectory;
    std::vector<StateType> path;
    StateType position;
    trajectory.push_back(state);  // Store initial condition

    // Allocate storage for intermediate calculations
    StateType k1(state.size());         // Slope at beginning of interval
    StateType k2(state.size());         // Slope at midpoint using k1
    StateType k3(state.size());         // Slope at midpoint using k2
    StateType k4(state.size());         // Slope at end of interval
    StateType tempState(state.size());  // Temporary state for intermediate evaluations

    // Main integration loop - continue while stop condition is true
    while (stopCondition(state)) {
        // Step 1: Compute k1 = f(t_n, y_n) - slope at start of interval
        derivatives(state, k1, 0.0);

        // Step 2: Compute k2 = f(t_n + h/2, y_n + h*k1/2) - slope at midpoint using k1
        for (size_t i = 0; i < state.size(); ++i) {
            tempState[i] = state[i] + 0.5 * timeStep * k1[i];
        }
        derivatives(tempState, k2, 0.5 * timeStep);

        // Step 3: Compute k3 = f(t_n + h/2, y_n + h*k2/2) - slope at midpoint using k2
        for (size_t i = 0; i < state.size(); ++i) {
            tempState[i] = state[i] + 0.5 * timeStep * k2[i];
        }
        derivatives(tempState, k3, 0.5 * timeStep);

        // Step 4: Compute k4 = f(t_n + h, y_n + h*k3) - slope at end of interval
        for (size_t i = 0; i < state.size(); ++i) {
            tempState[i] = state[i] + timeStep * k3[i];
        }
        derivatives(tempState, k4, timeStep);

        // Update state using weighted average: y_{n+1} = y_n + (h/6)(k1 + 2k2 + 2k3 + k4)
        // Weights: 1/6 for endpoints (k1, k4) and 2/6 for midpoints (k2, k3)
        for (size_t i = 0; i < state.size(); ++i) {
            state[i] += (timeStep / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        trajectory.push_back(state);  // Store updated state
        // extract position for path
        position.clear();
        position.push_back(state[0]);  // time
        for (size_t i = 1; i < state.size(); i += 6) {
            position.push_back(state[i]);  // positions only
            position.push_back(state[i + 1]);
            position.push_back(state[i + 2]);
        }
        path.push_back(position);
    }

    return path;
}

// Limitations and error estimates for RK4:
// - Fixed time step can lead to instability if too large.
// - Global error is O(h^4), but local truncation error is O(h^5).
// - Not suitable for stiff equations without modification.
// - Does not conserve energy in Hamiltonian systems.

/**
 * @brief Euler-Cromer (semi-implicit Euler) integration method
 *
 * A symplectic integrator that's particularly effective for oscillatory systems.
 * Updates velocity first, then uses the new velocity to update position.
 * This ordering preserves energy better than explicit Euler for Hamiltonian systems.
 *
 * For a state vector [time, pos1, vel1, pos2, vel2, ...]:
 * - v_{n+1} = v_n + a_n * h  (update velocity using current acceleration)
 * - x_{n+1} = x_n + v_{n+1} * h  (update position using NEW velocity)
 *
 * @param state Initial state vector (modified in-place during integration)
 * @param derivatives Function that computes dy/dt given (y, dy/dt, t)
 * @param stopCondition Function returning true to continue, false to stop
 * @param timeStep Integration step size (can use larger steps than RK4)
 * @return Trajectory of state vectors at each time step
 */
template <typename StateType = std::vector<double>>
std::vector<StateType> euler_chromer(
    StateType& state, std::function<void(const StateType&, StateType&, double)> derivatives,
    std::function<bool(const StateType&)> stopCondition, double timeStep) {
    //
    std::vector<StateType> trajectory;
    trajectory.push_back(state);  // Store initial condition

    StateType derivs(state.size());  // Storage for derivatives

    // Main integration loop - continue while stop condition is true
    while (stopCondition(state)) {
        // Compute all derivatives at current state
        derivatives(state, derivs, 0.0);

        // CRITICAL: Update velocity components FIRST using current accelerations
        // This is the "Cromer" modification that makes the method semi-implicit
        // For state vector: [time, pos1, vel1, pos2, vel2, ...]
        // Velocity components are at even indices >= 2
        for (size_t i = 2; i < state.size(); i += 2) {
            state[i] += derivs[i] * timeStep;
        }

        // Then update position components using the NEWLY COMPUTED velocities
        // This ordering preserves symplectic structure and improves stability
        // Position components are at odd indices >= 1
        for (size_t i = 1; i < state.size(); i += 2) {
            state[i] += state[i + 1] * timeStep;
        }

        // Update time (state[0]) - always advances at constant rate
        state[0] += timeStep;

        trajectory.push_back(state);
    }
    return trajectory;
}

// Limitations and error estimates for Euler-Cromer:
// - Simple and easy to implement.
// - First-order method with global error O(h).
// - More stable than standard Euler for oscillatory systems.
// - Does not conserve energy in Hamiltonian systems over long times.
// - Not suitable for stiff equations.
// - Accuracy can be low even for small time steps.

/**
 * @brief Verlet integration method for conservative systems (position-only format)
 *
 * A symplectic integrator that is particularly well-suited for conservative systems
 * such as planetary motion. Uses only positions in the state vector - velocities are
 * derived from the position history.
 *
 * State vector format: [time, x1, y1, z1, x2, y2, z2, ..., xN, yN, zN]
 * - state[0] = time
 * - state[1..3] = position of(x, y, z)
 * - state[4..6] = position of object 2 (x, y, z)
 * - etc.
 *
 * Algorithm per step (applied to each position component):
 * - x_{n+1} = 2x_{n} - x_{n-1} + a_{n} * h^2
 *
 * The acceleration function receives positions and must compute accelerations
 * based on those positions alone.
 *
 * @param state Initial state vector [time, x1, y1, z1, x2, y2, z2, ...]
 * @param previousState Previous positions for Verlet algorithm
 * @param accelerationFunc Function that computes accelerations from positions
 * @param stopCondition Function returning true to continue, false to stop
 * @param timeStep Integration step size
 * @return Trajectory of state vectors at each time step
 */
template <typename StateType = std::vector<double>>
std::vector<StateType> verlet(StateType& state, StateType& previousState,
                              std::function<void(const StateType&, StateType&)> accelerationFunc,
                              std::function<bool(const StateType&)> stopCondition,
                              double timeStep) {
    //
    std::vector<StateType> trajectory;
    trajectory.push_back(state);  // Store initial condition

    StateType accelerations(state.size());  // Storage for accelerations

    // Main integration loop - continue while stop condition is true
    while (stopCondition(state)) {
        // Compute accelerations at current positions
        accelerationFunc(state, accelerations);

        // Standard Verlet integration for all position components
        // x_{n+1} = 2x_n - x_{n-1} + a_n * h^2
        // Position components start at index 1 (index 0 is time)
        for (size_t i = 1; i < state.size(); ++i) {
            double newPosition =
                2.0 * state[i] - previousState[i] + accelerations[i] * timeStep * timeStep;

            // Store current position before updating
            previousState[i] = state[i];
            state[i] = newPosition;
        }

        // Update time (state[0]) - always advances at constant rate
        state[0] += timeStep;
        previousState[0] = state[0];

        trajectory.push_back(state);
    }
    return trajectory;
}

/**
 * @brief Initializes previous state for Verlet integration using RK4 bootstrapping
 *
 * Uses a single backward RK4 step to compute x_{-1} from initial conditions.
 * This provides higher accuracy than the simple Taylor expansion approach.
 *
 * @param positions Current positions [time, x1, y1, z1, x2, y2, z2, ...]
 * @param velocities Velocities [vx1, vy1, vz1, vx2, vy2, vz2, ...] (no time component)
 * @param accelerationFunc Function to compute accelerations from positions
 * @param timeStep Integration step size
 * @return Previous state vector for use with verlet()
 */
template <typename StateType = std::vector<double>>
StateType initializeVerletPreviousState(
    const StateType& positions, const std::vector<double>& velocities,
    std::function<void(const StateType&, StateType&)> accelerationFunc, double timeStep) {
    //
    // Create a combined state vector [time, pos1, vel1, pos2, vel2, ...] for RK4
    // positions: [time, x1, y1, z1, x2, y2, z2, ...]
    // velocities: [vx1, vy1, vz1, vx2, vy2, vz2, ...]
    size_t numPositions = positions.size() - 1;  // Exclude time
    StateType combinedState(1 + 2 * numPositions);

    combinedState[0] = positions[0];  // time
    for (size_t i = 0; i < numPositions; ++i) {
        combinedState[1 + 2 * i] = positions[i + 1];  // position
        combinedState[2 + 2 * i] = velocities[i];     // velocity
    }

    // Create derivatives function for RK4 that works with combined state
    auto derivatives = [&accelerationFunc, numPositions](const StateType& state, StateType& derivs,
                                                         double /*dt*/) {
        // Extract positions for acceleration calculation
        StateType posState(numPositions + 1);
        posState[0] = state[0];
        for (size_t i = 0; i < numPositions; ++i) {
            posState[i + 1] = state[1 + 2 * i];
        }

        StateType accels(numPositions + 1);
        accelerationFunc(posState, accels);

        // derivs: [dt/dt, dx/dt=v, dv/dt=a, ...]
        derivs[0] = 1.0;  // time derivative
        for (size_t i = 0; i < numPositions; ++i) {
            derivs[1 + 2 * i] = state[2 + 2 * i];  // dx/dt = v
            derivs[2 + 2 * i] = accels[i + 1];     // dv/dt = a
        }
    };

    // Perform single backward RK4 step (negative timestep)
    StateType k1(combinedState.size());
    StateType k2(combinedState.size());
    StateType k3(combinedState.size());
    StateType k4(combinedState.size());
    StateType tempState(combinedState.size());
    double h = -timeStep;  // Negative for backward step

    derivatives(combinedState, k1, 0.0);

    for (size_t i = 0; i < combinedState.size(); ++i) {
        tempState[i] = combinedState[i] + 0.5 * h * k1[i];
    }
    derivatives(tempState, k2, 0.5 * h);

    for (size_t i = 0; i < combinedState.size(); ++i) {
        tempState[i] = combinedState[i] + 0.5 * h * k2[i];
    }
    derivatives(tempState, k3, 0.5 * h);

    for (size_t i = 0; i < combinedState.size(); ++i) {
        tempState[i] = combinedState[i] + h * k3[i];
    }
    derivatives(tempState, k4, h);

    for (size_t i = 0; i < combinedState.size(); ++i) {
        combinedState[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }

    // Extract previous positions from combined state
    StateType previousState(positions.size());
    previousState[0] = combinedState[0];  // time
    for (size_t i = 0; i < numPositions; ++i) {
        previousState[i + 1] = combinedState[1 + 2 * i];  // positions only
    }

    return previousState;
}

// Limitations and error estimates for Verlet:
// - Second-order method with global error O(h^2).
// - Excellent energy conservation for conservative systems.
// - Not self-starting; requires special handling for the first step.
// - Not suitable for non-conservative forces without modification.

}  // namespace integrator
