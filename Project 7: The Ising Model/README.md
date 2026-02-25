# Project 7: The 3D Ising Model

**Author:** Nels Buhrley
**Language:** C++17 with OpenMP · Python 3 (visualization)
**Build:** `make release` — see [Build & Run](#build--run)

---

## Overview

This project implements a full **Monte Carlo simulation of the three-dimensional Ising model** on a cubic lattice. The simulation sweeps across a two-dimensional parameter space of temperature $T$ and external magnetic field $h$, producing a surface of average magnetization $\langle m \rangle(T, h)$ that captures the system's full thermodynamic behavior — including the **ferromagnetic phase transition** and the **onset of spontaneous symmetry breaking**.

The implementation emphasizes both physical correctness and computational performance, combining the **Metropolis–Hastings algorithm**, a **checkerboard (black-red) lattice decomposition**, **precomputed energy lookup tables**, and **OpenMP multi-threaded parallelism** to efficiently map the enormous parameter space.

---

## Physics Background

### The Ising Hamiltonian

The Ising model is one of the foundational models of statistical mechanics. It places discrete binary spins $\sigma_i \in \{-1, +1\}$ on the sites of a lattice — here a three-dimensional cubic lattice of $N \times N \times N$ sites — and defines their total energy through the Hamiltonian:

$$\mathcal{H} = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i$$

where:
- $J > 0$ is the **exchange coupling constant** (ferromagnetic in this implementation, set to 1 in natural units)
- the first sum $\langle i, j \rangle$ runs over all **nearest-neighbor pairs** (each 3D site has exactly 6 neighbors)
- $h$ is the **external magnetic field**, which biases the system toward spin alignment
- the second sum runs over all $N^3$ spins

### Phase Transitions and Critical Behavior

At low temperature and zero field, the system exhibits **spontaneous magnetization**: thermal fluctuations are insufficient to overcome the exchange energy, and the spins align collectively into a ferromagnetic phase. As temperature rises, entropy dominates and the magnetization collapses continuously toward zero at a **critical temperature** $T_c$ — the **Curie point**.

This **second-order phase transition** is characterized by:

- **Diverging correlation length** $\xi \sim |T - T_c|^{-\nu}$, meaning spins become correlated over arbitrarily large distances near $T_c$
- **Power-law scaling** of the order parameter: $\langle m \rangle \sim (T_c - T)^\beta$ for $T < T_c$
- **Critical slowing down** of Monte Carlo dynamics near $T_c$, as large correlated domains must reorganize

The 3D Ising model (unlike the exactly solvable 2D case due to Onsager) has a **Curie temperature** of approximately:

$$T_c \approx 4.51 \; J/k_B$$

in units where $J = k_B = 1$. Observing this transition emerge naturally from the simulation — with no closed-form solution guiding it — is one of the most compelling demonstrations of the power of Monte Carlo methods.

### Observable: Average Magnetization

The primary observable computed at each $(T, h)$ point is the **average magnetization per spin**:

$$\langle m \rangle = \frac{1}{N^3} \sum_i \sigma_i$$

Scanning $\langle m \rangle$ across the full $(T, h)$ plane reveals the **equation of state** of the model, including:
- The ferromagnetic lobes at low $T$ and small $|h|$ where $\langle m \rangle \to \pm 1$
- The sharp discontinuity in $\langle m \rangle(h)$ at $T < T_c$ as $h$ crosses zero (first-order transition in $h$)
- The smooth, paramagnetic response at $T > T_c$

---

## Algorithmic Design

### Metropolis–Hastings Algorithm

The simulation uses the **Metropolis–Hastings Markov Chain Monte Carlo (MCMC)** method to sample the Boltzmann distribution:

$$P(\{\sigma\}) \propto e^{-\mathcal{H}/k_B T}$$

At each Monte Carlo step, a candidate spin flip $\sigma_i \to -\sigma_i$ is proposed. The energy change from flipping spin $i$ at local field $h$ depends only on the six nearest neighbors:

$$\Delta E = 2\sigma_i \left(\sum_{j \in \text{nn}(i)} \sigma_j + h\right)$$

The Metropolis acceptance rule then gives:

$$P_{\text{accept}} = \begin{cases} 1 & \text{if } \Delta E \leq 0 \\ e^{-\Delta E / T} & \text{if } \Delta E > 0 \end{cases}$$

This guarantees that the Markov chain satisfies **detailed balance**, so the long-time stationary distribution converges to the true thermal equilibrium of the system.

### Precomputed Energy Lookup Tables

A key optimization targets `flipSpin()`, the innermost function called $O(N^3 \times \text{iterations})$ times. In a 3D model with 6 neighbors, the neighbor sum $S = \sum_{j \in \text{nn}} \sigma_j$ is restricted to the discrete set $\{-6, -4, -2, 0, +2, +4, +6\}$ — only **7 values**. Combined with the 2 possible spin states ($\sigma_i = \pm 1$), only **14 distinct $\Delta E$ values** are ever possible.

Rather than computing $e^{-\Delta E/T}$ inside the inner loop (which involves a costly floating-point exponential operation), both $\Delta E$ and $e^{-\Delta E / T}$ are **precomputed once** into small 2×7 tables at the start of each simulation:

```cpp
float deltaE_table[2][7];   // ΔE for each (spin, neighbor-sum) combination
float exp_table[2][7];      // exp(-ΔE/T) for each combination
```

During the spin update, a neighbor sum is computed, mapped to a table index, and the acceptance probability is read with a simple array lookup — eliminating all redundant floating-point transcendental calls from the critical path. This is a well-known technique from the computational physics literature that yields substantial speedups at no loss of accuracy.

### Checkerboard (Black-Red) Decomposition

To eliminate **locational biasies** during spin updates while preserving the ability to parallelize within a sweep, this implementation uses a **checkerboard decomposition** — also known as the black-red or odd-even subgraph partition.

The 3D cubic lattice is colored in two alternating sublattices:
- **Black sites**: all $(x, y, z)$ with $(x + y + z)$ even
- **Red sites**: all $(x, y, z)$ with $(x + y + z)$ odd

Because every neighbor of a black site is red, and vice versa, all black sites can be updated **simultaneously without conflict**: no two updated spins share a neighbor. The sweep performs a full black pass followed by a full red pass, completing one lattice-wide Monte Carlo step while maintaining the correct update statistics.

In the code, this is implemented by striding the inner $z$-loop by 2, selecting the correct parity based on $(x + y) \% 2$:

```cpp
// Black pass
for (z = (x + y) % 2 + 1; z < n - 1; z += 2)  flipSpin(x, y, z);
// Red pass
for (z = (x + y + 1) % 2 + 1; z < n - 1; z += 2)  flipSpin(x, y, z);
```

### Periodic Boundary Conditions via Ghost Layers

To simulate a bulk material without artificial surface effects, the lattice uses **periodic boundary conditions** (PBCs). Rather than wrapping indices on every access — which introduces branch overhead in the inner loop — the lattice is allocated with **size $(N+2)^3$**, adding one layer of **ghost cells** on each face.

Before each sweep, the ghost layers are explicitly refreshed by copying the opposite interior face:

```cpp
setSpin(x, y, 0,   getSpin(x, y, n-2));  // z-bottom ghost ← z-top interior
setSpin(x, y, n-1, getSpin(x, y, 1));    // z-top ghost    ← z-bottom interior
// ... similarly for x and y faces
```

This keeps the inner loop — which reads from ghost cells — branch-free and cache-friendly, at the cost of a brief $O(N^2)$ boundary refresh per sweep, which is negligible compared to the $O(N^3)$ inner work.

### Memory Layout

Spins are stored as `int8_t` (one signed byte each) in a flat 1D `std::vector<int8_t>` using row-major (C-order) indexing:

$$\text{index}(x, y, z) = x \cdot N^2 + y \cdot N + z$$

This representation is maximally cache-efficient for the sequential inner $z$-loop, ensuring that consecutive spin accesses fall in adjacent memory locations. Using `int8_t` rather than `int` reduces the lattice memory footprint by a factor of 4, keeping a 100×100×100 lattice well within L3 cache.

---

## Parallelization Strategy

The full simulation sweeps a **200 × 200 grid** of $(T, h)$ points — 40,000 independent Material instances — each run for 200 full lattice sweeps. The outer parameter sweep is trivially parallel: each $(T, h)$ pair is a completely independent simulation with no shared mutable state.

### OpenMP Collapse + Dynamic Scheduling

The outer loops are parallelized with OpenMP:

```cpp
#pragma omp parallel for collapse(2) schedule(dynamic)
for (size_t i = 0; i < temps.size(); i++) {
    for (size_t j = 0; j < h_values.size(); j++) {
        Material material(n, temps[i], h_values[j], iterations, thread_seeds[i]);
        material.runSimulation();
        avg_magnetizations[j][i] = material.getAverageMagnetization();
    }
}
```

- **`collapse(2)`** flattens the two-dimensional $(T, h)$ loop into a single iteration space of 40,000 tasks, allowing the OpenMP runtime to distribute work evenly across all available cores without manual tiling.
- **`schedule(dynamic)`** enables adaptive load distribution. Simulations near the critical temperature $T_c$ exhibit **critical slowing down** — Monte Carlo chains mix more slowly, so nearby $(T, h)$ points may take longer. Dynamic scheduling handles this variance automatically by assigning new tasks to idle threads as they finish.

### Thread-Safe Random Number Generation

Each Material object maintains its own **Mersenne Twister** (`std::mt19937`) instance, seeded independently. A single-threaded **master RNG** generates unique, cryptographically determined seeds from `std::random_device` before the parallel region launches:

```cpp
std::random_device rd;
std::mt19937 master_gen(rd());
std::uniform_int_distribution<uint32_t> seed_dist;

for (size_t i = 0; i < temps.size(); i++)
    thread_seeds[i] = seed_dist(master_gen);
```

This avoids two common parallelism pitfalls:
1. **Seed correlation** — naively seeding with `i` or `omp_get_thread_num()` produces correlated random streams, biasing results near the phase boundary
2. **Data races** — sharing a single RNG across threads without locking corrupts the generator's internal state

Each thread owns its RNG exclusively; no mutexes or atomic operations are needed in the hot loop.

---

## Sources of Error and Limitations

| Source | Nature | Mitigation |
|---|---|---|
| **Statistical fluctuations** | Monte Carlo results are stochastic; each run differs slightly | Increase `iterations`; ensemble-average over multiple seeds |
| **Finite-size effects** | A finite $N$ smears the phase transition; $T_c(N)$ differs from the thermodynamic limit $T_c(\infty)$ | Increase $N$; apply finite-size scaling analysis ($\xi \sim N$ at $T_c$) |
| **Equilibration error** | Early sweeps retain memory of the initial state (random or uniform) | Discard a thermalization burn-in period before accumulating observables |
| **Discretization of parameter space** | $T$ and $h$ are sampled on finite grids | Increase `tempSteps` / `numHSteps` in regions of interest |
| **Model simplification** | The model uses $J = k_B = 1$, nearest-neighbor coupling only, and classical spins | Extend to longer-range or anisotropic couplings for more realistic materials |

**Computational complexity** (per $(T, h)$ point):

$$\mathcal{O}\!\left(N^3 \times \text{iterations}\right)$$

**Total simulation complexity:**

$$\mathcal{O}\!\left(N^3 \times \text{iterations} \times N_T \times N_h\right)$$

For the default parameters ($N = 100$, iterations $= 200$, $N_T = N_h = 200$):
$$\sim 8 \times 10^{10} \text{ spin-update operations}$$

This is only tractable at the chosen scale because of the combined effect of precomputed lookup tables, checkerboard parallelism within each sweep, and OpenMP multi-core parallelism across the parameter space.

---

## Build & Run

### Prerequisites

- **C++17** compatible compiler (`g++` or `clang++`)
- **OpenMP** (via Homebrew: `brew install libomp` on macOS)
- **zlib** (for `.npz` file output)
- **Python 3** with `numpy` and `matplotlib` (for visualization)

### Build

```bash
# Standard optimized build
make release

# Aggressive optimization (may introduce minor floating-point drift)
make unsafe

# Debug build (no OpenMP, full warnings)
make debug

# Profile-guided optimization (two-step)
make profile-gen && ./bin/main && make profile-use
```

### Run

```bash
./bin/main
```

Output is saved to `output/ising_results.npz` as three NumPy arrays:

| Array | Shape | Description |
|---|---|---|
| `avg_magnetizations` | `(N_h, N_T)` | Average magnetization $\langle m \rangle$ at each $(h, T)$ point |
| `temperatures` | `(N_T,)` | Temperature grid |
| `magnetic_fields` | `(N_h,)` | Magnetic field grid |

### Visualize

```bash
python3 plotting.py
```

Produces:
- **`output/magnetization_3d_surface_angle[1-4].png`** — 3D surface plots of $\langle m \rangle(T, h)$ from four azimuthal angles
- **`output/magnetization_contour.png`** — 2D contour map of the magnetization surface

---

## Simulation Parameters

Configured in [main.cpp](main.cpp):

| Parameter | Default | Description |
|---|---|---|
| `N` | 100 | Cubic lattice edge length ($N^3$ spins total) |
| `iterations` | 200 | Monte Carlo sweeps per $(T, h)$ point |
| `minTemp` / `maxTemp` | 0 – 45 | Temperature range (units: $J/k_B$) |
| `tempSteps` | 200 | Number of temperature grid points |
| `hMin` / `hMax` | −15 – +15 | External field range |
| `numHSteps` | 200 | Number of field grid points |

---

## Project Structure

```
Project 7: The Ising Model/
├── main.cpp          # Entry point: sets simulation parameters and calls runIsingSimulation()
├── processing.h      # Material class, Metropolis engine, parallelized sweep, NPZ output
├── Makefile          # Multi-target build: debug, release, unsafe, profile-guided
├── plotting.py       # Python visualization: 3D surface and 2D contour plots
└── output/
    └── ising_results.npz   # Simulation output (NumPy archive)
```

---

## Key Techniques at a Glance

| Technique | Purpose |
|---|---|
| Metropolis–Hastings MCMC | Correct Boltzmann sampling of spin configurations |
| Precomputed $\Delta E$ and $e^{-\Delta E/T}$ tables | Eliminates exponential calls from the inner loop |
| Checkerboard (black-red) decomposition | Race-free simultaneous updates within a sweep |
| Ghost boundary layers | Branch-free periodic boundary condition enforcement |
| `int8_t` spin storage + flat 1D array | Cache-efficient memory layout ($4\times$ smaller than `int`) |
| OpenMP `collapse(2)` + `dynamic` scheduling | Scalable multi-core parallelism across the $(T, h)$ parameter space |
| Per-thread Mersenne Twister with master-seeded seeds | Statistically independent, uncorrelated random streams |
| Profile-guided optimization (PGO) | Compiler uses runtime profiling data for branch prediction and inlining |

---

*Nels Buhrley — Computational Physics, 2026*
