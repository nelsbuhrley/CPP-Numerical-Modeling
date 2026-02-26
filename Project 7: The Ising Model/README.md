# Project 7: The 3D Ising Model

**Author:** Nels Buhrley
**Language:** C++17 with OpenMP · Python 3 (visualization)
**Build:** `make release` — see [Build & Run](#build--run)

---

## Overview

This project implements a full **Monte Carlo simulation of the three-dimensional Ising model** on a cubic lattice. The simulation sweeps across a two-dimensional parameter space of temperature $T$ and external magnetic field $h$, producing a surface of average magnetization $\langle m \rangle(T, h)$ that captures the system's full thermodynamic behavior — including the **ferromagnetic phase transition** and the **onset of spontaneous symmetry breaking**.

The implementation combines the **Metropolis–Hastings algorithm**, a **checkerboard (black-red) lattice decomposition**, **precomputed energy lookup tables**, and **OpenMP multi-threaded parallelism** to efficiently map the parameter space.

---

## Physics Background

### The Ising Hamiltonian

The Ising model places discrete binary spins $\sigma_i \in \{-1, +1\}$ on the sites of a 3D cubic lattice of $N \times N \times N$ sites, with total energy:

$$\mathcal{H} = -J \sum_{\langle i,j \rangle} \sigma_i \sigma_j - h \sum_i \sigma_i$$

where $J > 0$ is the ferromagnetic exchange coupling (set to 1 in natural units), the first sum runs over nearest-neighbor pairs (6 per site in 3D), and $h$ is the external magnetic field.

### Phase Transition

At low temperature and zero field the system exhibits **spontaneous magnetization**. As temperature rises, entropy dominates and magnetization collapses continuously to zero at the **Curie point** $T_c$. The 3D Ising model has:

$$T_c \approx 4.51 \; J/k_B$$

in units where $J = k_B = 1$.

### Observable: Average Magnetization

The primary output at each $(T, h)$ point is the **average magnetization per spin**:

$$\langle m \rangle = \frac{1}{N^3} \sum_i \sigma_i$$

---

## Code Structure

All simulation logic lives in two files:

| File | Role |
|---|---|
| `main.cpp` | Sets parameters and calls `runIsingSimulation()` |
| `processing.h` | `Material` class, Metropolis engine, parallel sweep, NPZ output |

### `Material` Class

Each `Material` instance represents one independent lattice at a fixed $(T, h)$ pair.

#### Construction

```cpp
Material(int n, float temperature, float magnetization, int numIterations, uint32_t seed)
```

The physical lattice is $N \times N \times N$ but the internal allocation is $(N+2)^3$ to hold **ghost boundary layers**. The constructor calls three setup methods in order:

1. `establishRNG()` — seeds a per-object `std::mt19937` from `seed`, giving each parallel worker its own independent random stream
2. `initializeSpinsRandomly()` — fills the inner lattice with random $\pm 1$ values; an overload `initializeSpinsUniformly(int8_t)` sets all spins to a single value instead
3. `precalculateEnergyTables()` — builds `deltaE_table[2][7]` and `exp_table[2][7]` (see [Precomputed Tables](#precomputed-energy-lookup-tables))

Spins are stored as `int8_t` in a flat 1D `std::vector<int8_t>` using row-major indexing:

$$\text{index}(x, y, z) = x \cdot N^2 + y \cdot N + z$$

Using `int8_t` rather than `int` keeps the $100^3$ lattice $4\times$ smaller and cache-friendly for the sequential $z$-loop. `getSpin()` and `setSpin()` are inlined accessors over this flat array.

---

#### Precomputed Energy Lookup Tables

`flipSpin()` is the innermost function, called $O(N^3 \times \text{iterations})$ times. In a 3D model with 6 neighbors, the neighbor sum $S \in \{-6,-4,-2,0,+2,+4,+6\}$ admits only **7 values**. Combined with the 2 spin states, only **14 distinct $\Delta E$ values** ever arise:

$$\Delta E = 2\sigma_i(S + h)$$

`precalculateEnergyTables()` computes both $\Delta E$ and $e^{-\Delta E/T}$ for all 14 combinations once, before any sweep:

```cpp
float deltaE_table[2][7];
float exp_table[2][7];
```

The inner loop then performs a plain array dereference instead of calling `exp()`.

---

#### `flipSpin(x, y, z)` — Metropolis Step

For site $(x,y,z)$, the neighbor sum is computed and mapped to a table index:

```cpp
void flipSpin(int x, int y, int z) {
    uint8_t neighborstate = (sum of 6 neighbors) / 2 + 3;  // maps [-6,6] → [0,6]
    uint8_t spinState = (getSpin(x, y, z) + 1) / 2;        // maps {-1,+1} → {0,1}
    if (deltaE_table[spinState][neighborstate] <= 0 ||
        distribution(gen) < exp_table[spinState][neighborstate]) {
        setSpin(x, y, z, -getSpin(x, y, z));
    }
}
```

The Metropolis acceptance rule is satisfied: always flip if $\Delta E \leq 0$, otherwise flip with probability $e^{-\Delta E/T}$.

---

#### `iteration()` — One Full Lattice Sweep

Each call performs three stages.

**Stage 1 — Ghost layer refresh** (periodic boundary conditions):

Rather than wrapping indices on every neighbor access, the $(N+2)^3$ lattice carries one ghost cell on each face. Before each sweep all six faces are refreshed by copying the opposite interior face:

```cpp
// Z faces
for (x) for (y) {
    setSpin(x, y, 0,   getSpin(x, y, n-2));   // bottom ghost ← top interior
    setSpin(x, y, n-1, getSpin(x, y, 1));     // top ghost    ← bottom interior
}
// Y and X faces updated the same way
```

This is $O(N^2)$ per sweep, negligible against the $O(N^3)$ spin work, and keeps the inner loop branch-free.

**Stage 2 — Black pass** (sites where $(x+y+z)$ is even):

```cpp
for (x = 1; x < n-1; x++)
    for (y = 1; y < n-1; y++)
        for (z = (x+y)%2 + 1; z < n-1; z += 2)
            flipSpin(x, y, z);
```

**Stage 3 — Red pass** (sites where $(x+y+z)$ is odd):

```cpp
for (x = 1; x < n-1; x++)
    for (y = 1; y < n-1; y++)
        for (z = (x+y+1)%2 + 1; z < n-1; z += 2)
            flipSpin(x, y, z);
```

The parity offset `(x+y)%2` selects the correct starting $z$ so every visited site satisfies the desired color. Striding by 2 ensures no two sites in the same pass share a neighbor, making all updates within a pass **conflict-free**. A black pass followed by a red pass constitutes one full Monte Carlo sweep.

---

#### `runSimulation()` and `getAverageMagnetization()`

```cpp
void runSimulation() {
    for (int i = 0; i < numIterations; i++) iteration();
}
```

After equilibrating, `getAverageMagnetization()` iterates over the inner lattice (indices 1 to $N$) and returns $\sum \sigma_i / N^3$.

---

### Simulation Pipeline: `runIsingSimulation()`

This free function in `processing.h` orchestrates the full parameter sweep:

```
1. Build temperature grid:  temps[i]    = tempMin + i * tempStep
2. Build field grid:        h_values[j] = hMin    + j * hStep
3. Seed master RNG from std::random_device
4. Generate thread_seeds[i] — one unique seed per temperature row
5. Allocate avg_magnetizations[numH][numT]
6. #pragma omp parallel for collapse(2) schedule(dynamic)
      for each (T, h) pair:
          construct Material → runSimulation() → store avg magnetization
7. saveResultsToNPZ(avg_magnetizations, temps, h_values, filename)
```

`collapse(2)` flattens the 200×200 grid into 40,000 independent tasks. `schedule(dynamic)` handles load imbalance near $T_c$ where critical slowing down makes some tasks longer. Seeds are generated from a single master RNG before the parallel region to avoid data races — each `Material` then owns its RNG exclusively with no locks needed in the hot loop.

---

## Sources of Error

| Source | Nature | Mitigation |
|---|---|---|
| Statistical fluctuations | MC results are stochastic | Increase `iterations`; ensemble-average over seeds |
| Finite-size effects | Finite $N$ smears the transition; $T_c(N) \neq T_c(\infty)$ | Increase $N$; apply finite-size scaling |
| Equilibration error | Early sweeps retain memory of the initial state | Discard a burn-in period before measuring |
| Parameter discretization | $T$ and $h$ sampled on finite grids | Increase `tempSteps`/`numHSteps` near $T_c$ |

**Computational complexity** (per $(T,h)$ point): $\mathcal{O}(N^3 \times \text{iterations})$

**Total**: $\mathcal{O}(N^3 \times \text{iterations} \times N_T \times N_h)$ — roughly $8 \times 10^{10}$ spin-update operations at default parameters.

---

## Build & Run

### Prerequisites

- **C++17** compiler (`g++` or `clang++`)
- **OpenMP** (`brew install libomp` on macOS)
- **zlib** (for `.npz` output)
- **Python 3** with `numpy` and `matplotlib`

### Build Targets

```bash
make release   # Optimized build (-O3, LTO, vectorization, march=native)
make unsafe    # Adds -ffast-math (may introduce minor FP drift)
make debug     # -O0, full warnings, OpenMP disabled
make profile-gen && ./bin/main && make profile-use  # Profile-guided optimization
```

### Run

```bash
./bin/main
```

Output is saved to `output/ising_results.npz`:

| Array | Shape | Description |
|---|---|---|
| `avg_magnetizations` | `(N_h, N_T)` | $\langle m \rangle$ at each $(h, T)$ point |
| `temperatures` | `(N_T,)` | Temperature grid |
| `magnetic_fields` | `(N_h,)` | Magnetic field grid |

### Visualize

```bash
python3 plotting.py
```

Produces 3D surface plots and a 2D contour map in `output/`.

---

## Simulation Parameters

Configured in [main.cpp](main.cpp):

| Parameter | Default | Description |
|---|---|---|
| `N` | 100 | Cubic lattice edge length ($N^3$ spins) |
| `iterations` | 200 | MC sweeps per $(T, h)$ point |
| `minTemp` / `maxTemp` | 0 – 45 | Temperature range ($J/k_B$) |
| `tempSteps` | 200 | Temperature grid points |
| `hMin` / `hMax` | −15 – +15 | External field range |
| `numHSteps` | 200 | Field grid points |

---

## Key Techniques

| Technique | Purpose |
|---|---|
| Metropolis–Hastings MCMC | Correct Boltzmann sampling |
| Precomputed $\Delta E$ and $e^{-\Delta E/T}$ tables | Eliminates `exp()` from the inner loop |
| Checkerboard (black-red) decomposition | Race-free simultaneous updates within a sweep |
| Ghost boundary layers | Branch-free periodic boundary enforcement |
| `int8_t` + flat 1D array | $4\times$ smaller than `int`; cache-friendly $z$-loop |
| `collapse(2)` + `dynamic` scheduling | Scalable multi-core parallelism across $(T, h)$ |
| Per-thread Mersenne Twister | Statistically independent, uncorrelated random streams |
| Profile-guided optimization (PGO) | Compiler uses runtime data for branch prediction and inlining |

---

*Nels Buhrley — Computational Physics, 2026*
