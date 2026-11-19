<p align="center">
  <a href="https://nels.buhrley.com/CPP-Numerical-Modeling/#academic-projects"><strong>← Back to Portfolio Hub</strong></a>
</p>

# Project 8: Molecular Dynamics — Capstone

> [!TIP]
> Every highlighted link in this page is clickable.
> For fast navigation use [Table of Contents](#table-of-contents)


**Author:** Nels Buhrley
**Language:** C++17 with OpenMP · Python 3 (visualization)
**HPC:** Run on the BYU Supercomputer (8 CPUs via SLURM)
**Build:** `make release` — see [Build & Run](#build--run)

---

## Snapshot

- Built a full 2D molecular dynamics engine in C++17 from scratch (Lennard-Jones physics, velocity Verlet integrator, configurable boundary conditions, gravity, and scripted heating/cooling).
- Solved a non-trivial parallelization problem in an $O(N^2)$ force kernel using OpenMP thread-local accumulators plus guided scheduling to keep updates race-free and scalable.
- Shipped production-style outputs for analysis and demos: structured NumPy/CSV artifacts, automated energy diagnostics, and parallel-rendered MP4 animations.
- Designed for reproducible compute workflows: config-driven runs (`simulation_1.cfg` / `simulation_2.cfg`), release/debug/PGO build modes, benchmark auto-tuning, and SLURM execution via `job.sh`.
- Demonstrates end-to-end engineering: numerical methods, performance optimization, HPC deployment, and polished communication of results.

For fast verification, jump to [Results](#results), [Build & Run](#build--run), and [Simulation Parameters](#simulation-parameters).

---

## Table of Contents

- [Snapshot](#snapshot)
- [Overview](#overview)
- [Physics Background](#physics-background)
  - [The Lennard-Jones Potential](#the-lennard-jones-potential)
  - [Equations of Motion](#equations-of-motion)
  - [Velocity Verlet Integration](#velocity-verlet-integration)
  - [Observables](#observables)
- [Code Structure](#code-structure)
- [Parallelization Narrative](#parallelization-narrative)
  - [The O(N²) Challenge](#the-on²-challenge)
  - [Race Conditions in Force Accumulation](#race-conditions-in-force-accumulation)
  - [The Thread-Local Accumulator Solution](#the-thread-local-accumulator-solution)
  - [Guided Scheduling for Triangular Loops](#guided-scheduling-for-triangular-loops)
  - [Lessons Learned](#lessons-learned)
- [Visualization (`plotting.py`)](#visualization-plottingpy)
- [Results](#results)
- [Sources of Error](#sources-of-error)
- [Build & Run](#build--run)
- [Simulation Parameters](#simulation-parameters)
- [Key Techniques](#key-techniques)
- [Project Structure](#project-structure)

---

## Overview

This project implements a **2D Molecular Dynamics simulation** of interacting particles governed by the **Lennard-Jones potential**. The system evolves via the **velocity Verlet integration** algorithm with periodic boundary conditions, modeling the thermodynamic behavior of a simple fluid. An external **energy injection/extraction schedule** drives the system through heating and cooling phases, allowing observation of phase-like transitions in real time.

The implementation combines **OpenMP parallelism** for force calculations, **minimum image convention** for periodic boundaries, and a **cell-based Verlet neighbor list** to reduce unnecessary pair checks. Results are saved as `.npy` and `.csv` files, with Python tooling that produces parallel-rendered particle animations and comprehensive energy diagnostics.

---

## Physics Background

### The Lennard-Jones Potential

Particles interact via the Lennard-Jones (12-6) potential in reduced units ($\sigma = \varepsilon = m = 1$):

$$U(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]$$

The steep $r^{-12}$ repulsion models short-range Pauli exclusion, while the $r^{-6}$ attraction captures van der Waals dispersion forces. A cutoff radius $r_c = 2.5\,\sigma$ truncates the potential for computational efficiency — contributions beyond $r_c$ are negligible.

### Equations of Motion

Each particle obeys Newton's second law:

$$m\,\ddot{\mathbf{r}}_i = \mathbf{F}_i = -\nabla_i \sum_{j \neq i} U(r_{ij})$$

The force on particle $i$ due to particle $j$ is:

$$\mathbf{F}_{ij} = \frac{24\varepsilon}{r^2}\left[2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right]\mathbf{r}_{ij}$$

Newton's third law ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$) halves the number of pair evaluations.

### Velocity Verlet Integration

The simulation advances positions and velocities using the velocity Verlet scheme, which is time-reversible and symplectic:

$$\mathbf{v}_i\left(t + \tfrac{\Delta t}{2}\right) = \mathbf{v}_i(t) + \tfrac{\Delta t}{2}\,\mathbf{a}_i(t)$$

$$\mathbf{r}_i(t + \Delta t) = \mathbf{r}_i(t) + \mathbf{v}_i\left(t + \tfrac{\Delta t}{2}\right)\,\Delta t$$

$$\mathbf{v}_i(t + \Delta t) = \mathbf{v}_i\left(t + \tfrac{\Delta t}{2}\right) + \tfrac{\Delta t}{2}\,\mathbf{a}_i(t + \Delta t)$$

This two-stage velocity update requires force evaluation only once per time step.

### Observables

| Observable | Definition |
|---|---|
| Temperature | $T = \frac{1}{N d}\sum_i m v_i^2$ (equipartition theorem, $k_B = 1$) |
| Kinetic Energy | $K = \frac{1}{2}\sum_i m v_i^2$ |
| Potential Energy | $U = \sum_{i<j} U_\text{LJ}(r_{ij})$ |
| Total Energy | $E = K + U$ |

---

## Code Structure

All simulation logic lives in two files:

| File | Role |
|---|---|
| `main.cpp` | Parses config, runs simulation, and supports `--benchmark` tuning sweeps |
| `processing.h` | `MolucularSystem` class: integration, neighbor-list force evaluation, energy tracking, and I/O |

### `MolucularSystem` Class

Each `MolucularSystem` instance represents a complete simulation state for $N$ particles in a 2D periodic box.

#### Construction

```cpp
MolucularSystem(config)
```

- **Positions** are integrated in `currPositions` and sampled into `sampledPositions` every `stepSkip` steps to keep output sizes manageable.
- **Velocities** and **accelerations** are single-frame arrays of size `numParticles`.
- All particles start at rest except particle 0, which receives a small initial velocity `{0.001, 0.001}` to break symmetry and seed energy flow.
- The dimensionality is compile-time configurable via `static constexpr int d = 2`.

#### Constants (Reduced Units)

| Constant | Value | Description |
|---|---|---|
| `d` | 2 | Spatial dimensions |
| `rc` | 2.5 σ | Lennard-Jones cutoff radius |
| `rc2` | 6.25 σ² | Precomputed $r_c^2$ to avoid square roots |

---

#### `calculateAccelerations(t)` — Force Calculation

Computes Lennard-Jones forces using a Verlet neighbor list with linked-cell binning:

```cpp
if (shouldRebuildNeighborList(currPositions))
  rebuildNeighborList(currPositions);

for (p1 = 0; p1 < N; p1++)
  for (p2 in neighbors[p1])
    dx = p1.x - p2.x
    dx -= L * round(dx / L)      // minimum image convention
    r2 = dx*dx + dy*dy
    if (r2 > rc2) continue       // force cutoff
    invR2 = 1.0 / r2
    invR6 = invR2 * invR2 * invR2
    F = 24.0 * (2.0 * invR6 * invR6 - invR6) * invR2
    a[p1] += F;  a[p2] -= F      // Newton's third law
```

**Parallelism:** When `numParticles > 50`, the loop over particles is parallelized with OpenMP. Each thread accumulates into thread-local acceleration/potential buffers and the buffers are reduced after the parallel region.

**Minimum image convention:** Periodic boundary conditions are enforced by shifting displacements to the nearest image: `dx -= L * round(dx / L)`.

---

#### `verletStep(t)` — Velocity Verlet Integration

Each call advances the system by one time step $\Delta t$:

1. **Half-kick:** Update velocities by half a time step using current accelerations
2. **Drift:** Update positions using the half-stepped velocities
3. **Wrap:** Apply periodic boundary conditions via `pos -= L * floor(pos / L)`
4. **Recalculate forces:** Call `calculateAccelerations(t)` at the new positions
5. **Half-kick:** Complete the velocity update with the new accelerations
6. **Energy bookkeeping:** Call `energyCalculations(t)`

Both velocity loops are parallelized with `#pragma omp parallel for` when `numParticles > 100`.

---

#### `energyCalculations(t)` — Energy Tracking & Injection

At each time step, the method:

1. Checks if the current step matches the next entry in `energyFunction` — if so, **rescales all velocities** by a factor $\lambda$:

$$\lambda = \sqrt{1 + \frac{2\,\Delta E_\text{desired}}{2K}}$$

This uniformly scales kinetic energy to inject or remove a precise amount of energy.

2. Records temperature, potential energy, kinetic energy, and total energy for postprocessing.

---

#### Energy Schedule: `buildEnergyFunction()`

The energy injection/extraction schedule is:

| Phase | Time Steps | Action |
|---|---|---|
| 0 – 5% | Settle | No energy change |
| 5 – 50% | Heating | Inject energy at a constant rate |
| 50 – 95% | Cooling | Remove energy at a constant rate |
| 95 – 100% | Settle | No energy change |

This drives the system through a full heating-cooling cycle, allowing observation of melting and crystallization behavior.

---

#### `runSimulation()` — Main Loop

```cpp
calculateAccelerations(0);      // initial forces
energyCalculations(0);          // initial energy bookkeeping
for (t = 1; t < timeSteps; t++)
    verletStep(t);              // advance the system
```

Progress is logged every 5% of the simulation.

---

#### Output Methods

| Method | Output |
|---|---|
| `saveResultsToNpy()` | Separate `.npy` arrays for sampled positions, energies, temperatures, and metadata |
| `saveEnergyToCSV()` | CSV with columns: TimeStep, Temperature, PE, KE, TE |
| `savePositionsToCSV()` | CSV with columns: TimeStep, ParticleIndex, X, Y |
| `save()` | Auto-creates `output/out_N/` (incrementing index) and writes NPY + CSV outputs |

---

## Parallelization Narrative

### The O(N²) Challenge

The baseline all-pairs Lennard-Jones loop is $O(N^2)$, which becomes expensive quickly for dense systems and long trajectories. The current code reduces this cost by rebuilding a Verlet neighbor list only when particles move enough to invalidate the list.

This turns the hot loop into "iterate only nearby particles" instead of "iterate every particle pair," while preserving exact pairwise force updates inside the cutoff radius.

### Race Conditions in Force Accumulation

Race hazards still exist because Newton's third law updates two particles per interaction. The implementation avoids shared-write races by assigning each thread a private acceleration buffer:

```cpp
for (int p1 = 0; p1 < N; p1++)
  for (int p2 : neighborList[p1])
        F = compute_LJ_force(p1, p2);
    localAcceleration[p1] += F;
    localAcceleration[p2] -= F;
```

Each thread writes only to its own local arrays during force traversal; shared arrays are combined in a deterministic reduction pass after the parallel loop.

### The Thread-Local Accumulator Solution

The current solution combines two optimizations:

1. **Linked-cell neighbor-list rebuilds** with configurable `neighborSkin`.
2. **Thread-local force accumulation** with post-loop buffer summation.

This structure keeps force evaluation race-free while reducing pair checks significantly for realistic densities.

```cpp
#pragma omp parallel if (numParticles > 50)
{
    // Each thread writes to its own acceleration buffer.
    std::vector<std::array<double, d>> localAccelerations = threadAccelerations[tid];
    double localPE = 0.0;

    #pragma omp for schedule(guided)
    for (int p1 = 0; p1 < numParticles; p1++) {
        for (int p2 : neighborList[p1]) {
            // ... force calculation ...
            localAccelerations[p1][dim] += force_component;
            localAccelerations[p2][dim] -= force_component;  // safe: thread-private
        }
    }
}

  // After the parallel region, reduce thread buffers into shared accelerations.
```

  The trade-off is extra memory for per-thread acceleration arrays, but it removes fine-grained synchronization overhead and scales well on shared-memory nodes.

### Guided Scheduling for Triangular Loops

Even with neighbor lists, work per particle is non-uniform because local density varies. `schedule(guided)` is used so overloaded threads can shed work dynamically:

```cpp
#pragma omp for schedule(guided)
for (int p1 = 0; p1 < numParticles; p1++) { ... }
```

This improves balance for both dense startup lattices and later disordered states.

### Lessons Learned

1. **Neighbor lists are the biggest practical speedup.** Rebuilding occasionally is cheaper than checking all pairs each step.

2. **Thread-local buffers remove write races cleanly.** Newton's third-law updates stay local during parallel work and are reduced afterward.

3. **Guided scheduling remains important.** Density-driven imbalance appears even when pair loops are no longer strictly triangular.

4. **Auto-tuning matters on shared clusters.** The benchmark mode (`--benchmark`) sweeps threads and `neighborSkin` to select better run settings per node.

---

## Visualization (`plotting.py`)

The Python pipeline performs two tasks with separate scripts:

### 1. Particle Animation (Parallel Rendering)

The animation is rendered in parallel using `multiprocessing`:

1. Frame indices are split across configurable workers (`--workers`, default based on CPU count)
2. Each worker renders its segment to a temporary `.mp4` via `matplotlib.animation`
3. Segments are stitched with `ffmpeg -concat` into a single animation

This avoids the serial bottleneck of rendering thousands of frames on one core.

### 2. Energy Analysis Plots

`plot_energy.py` creates summary and diagnostic figures:

| Panel | Content |
|---|---|
| Composite Figure | Energy components, drift, temperature, and temperature-vs-energy scatter |
| Detailed Figure | Binned temperature-over-energy trend plus raw samples |

---

## Results

### Particle Animation

<p align="center">
  <video width="70%" controls autoplay muted loop style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
    <source src="https://nels.buhrley.com/CPP-Numerical-Modeling/assets/videos/md_animation.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <br>
  <em>Particle trajectory animation showing the heating-cooling cycle. Particles evolve from a regular grid through a gas-like disordered phase during heating, then re-order as energy is extracted, illustrating the connection between kinetic energy and temperature at the microscopic level.</em>
</p>

<!-- Uncomment below if you have an MP4 instead of GIF:
<p align="center">
  <video src="output/out_0/md_animation.mp4" width="70%" autoplay loop muted playsinline>
    Your browser does not support the video tag.
  </video>
</p>
-->

### Energy Analysis

The energy plot shows the heating/cooling cycle: kinetic energy rises during the injection phase, plateaus, and falls during extraction. Total energy drift remains bounded, validating the Verlet integrator's symplectic properties.

---

## Sources of Error

| Source | Nature | Mitigation |
|---|---|---|
| Time-step discretization | Finite $\Delta t$ introduces integration error | Use smaller `finalTime / timeSteps`; Verlet is $O(\Delta t^2)$ per step |
| Cutoff truncation | Ignoring interactions beyond $r_c$ introduces a systematic energy offset | $r_c = 2.5\sigma$ captures $>98\%$ of the LJ well; could add tail corrections |
| Finite-size effects | Small $N$ or small box $L$ affects thermodynamic behavior | Increase particle count and box size |
| Velocity rescaling | Instantaneous rescaling is not thermostatting — not canonical ensemble | Acceptable for driven heating/cooling; use Nosé-Hoover for equilibrium studies |
| Initial symmetry | Grid start with zero velocities is artificial | Small perturbation on particle 0 breaks symmetry; long heating phase equilibrates |

**Computational complexity:** Force evaluation is neighbor-list based: typical runtime is closer to $\mathcal{O}(N \times n_{nbr} \times T)$ with periodic $\mathcal{O}(N)$ rebuilds, while worst-case behavior remains $\mathcal{O}(N^2 \times T)$ in very dense/pathological layouts.

---

## Build & Run

### Prerequisites

- **C++17** compiler (`g++` or `clang++`)
- **OpenMP** (`brew install libomp` on macOS)
- **zlib** (for `cnpy` I/O linkage)
- **Python 3** with `numpy` and `matplotlib`
- **ffmpeg** (for animation rendering)

### Build Targets

```bash
make release   # Optimized build (-O3, LTO, vectorization, march=native)
make unsafe    # Adds -ffast-math (may introduce minor FP drift)
make debug     # -O0, full warnings, OpenMP disabled
make profile-gen && ./bin/main simulation_2.cfg && make profile-use  # Profile-guided optimization
```

### Run

```bash
./bin/main simulation_2.cfg
./bin/main simulation_2.cfg --benchmark
```

Output is auto-saved to `output/out_N/` (incrementing index):

| File | Description |
|---|---|
| `positions.npy` | Position history as `(sampledSteps, N, 2)` sampled every `stepSkip` solver steps |
| `temperatures.npy` | Temperature time series |
| `potentialEnergies.npy` | Potential energy time series |
| `kineticEnergies.npy` | Kinetic energy time series |
| `totalEnergies.npy` | Total energy time series |
| `metadata.npy` | `[width, height, numParticles, timeSteps, finalTime]` |
| `energy_data.csv` | Sampled energy/temperature CSV |
| `positions_data.csv` | Sampled per-particle trajectory CSV |

### Configuration (simulation_1.cfg / simulation_2.cfg)

The simulation is fully configured through `.cfg` files using `key=value` pairs.

Required keys currently parsed by `main.cpp`/`processing.h`:

| Key | Type | Meaning |
|---|---|---|
| `timeSteps` | int | Number of integration steps |
| `finalTime` | float | Physical end time; `dt = finalTime / timeSteps` |
| `width` | float | Box width |
| `height` | float | Box height |
| `xBoundaryCondition` | string | `periodic` or `reflective` |
| `yBoundaryCondition` | string | `periodic` or `reflective` |
| `gravity` | float | Constant acceleration in `-y` |
| `stepSkip` | int | Sampling stride for stored positions/CSV output |
| `neighborSkin` | float | Verlet neighbor-list skin distance |
| `showProgress` | int (optional) | Progress-print toggle (`0` off, `1` on; default on) |
| `initialPositionsInstructions` | string | Shape instructions for initial particle set |
| `EnergyInstructions` | string | Time-segmented heating/cooling schedule |

`initialPositionsInstructions` format:

```text
Shape,cx,cy,rotation,spacing,size[,...];Shape,cx,cy,rotation,spacing,size[,...]
```

- Supported shapes: `Rectangle`, `Hexagon`, `Rhombus`, `Triangle`, `Circle`
- Rectangle uses 7 fields: `Rectangle,cx,cy,rotation,spacing,numX,numY`
- Other shapes use 6 fields: `Shape,cx,cy,rotation,spacing,size`

`EnergyInstructions` format:

```text
startPercent,endPercent,totalEnergyChangePer100Particles;...
```

Each segment distributes its total energy change uniformly over that percent range.

### Visualize

```bash
python3 plotting.py --workers 8
python3 plot_energy.py --run 0
```

`plotting.py` automatically loads the newest `output/out_N/` directory and writes:

- `md_animation_N.mp4`

`plot_energy.py` automatically loads either the latest or requested run and writes:

- `energy_analysis_N.png`
- `temperature_over_energy_N.png`
- `energy_summary_N.csv`

Note: `plotting.py` currently points to a cluster-specific ffmpeg path. If running locally, update the `FFMPEG` variable or replace it with your system ffmpeg path.

### HPC (SLURM)

```bash
sbatch job.sh
```

The batch script builds in release mode only when needed, optionally auto-tunes `OMP_NUM_THREADS` and `neighborSkin` via benchmark mode, runs `./bin/main simulation_2.cfg`, then submits `anamation.sh` and `plotting.sh` as follow-up SLURM jobs.

---

## Simulation Parameters

Configured in `simulation_2.cfg` (not hardcoded in `main.cpp`).

| Parameter | Default | Description |
|---|---|---|
| `initialPositionsInstructions` | `Rectangle, 125.0, 30.0, 0.0, 1.1, 200, 50` | Initial particle layout (shape-composed) |
| `timeSteps` | `1000000` | Number of integration steps |
| `finalTime` | `2000.0` | Total simulation time (reduced units) |
| `width`,`height` | `250.0`,`150.0` | 2D box dimensions |
| `xBoundaryCondition`,`yBoundaryCondition` | `periodic`,`reflective` | Boundary handling by axis |
| `gravity` | `0.01` | Constant downward acceleration |
| `neighborSkin` | `0.2` | Skin distance for neighbor-list rebuild policy |
| `stepSkip` | `160` | Save sampled frames every 160 timesteps |
| `EnergyInstructions` | `2,8,-50;8,50,800;50,60,-600;60,95,-200` | Multi-stage heating/cooling schedule |
| `rc` | 2.5 σ | Lennard-Jones cutoff radius |

---

## Key Techniques

| Technique | Purpose |
|---|---|
| Velocity Verlet integration | Time-reversible, symplectic $O(\Delta t^2)$ integrator |
| Lennard-Jones potential with cutoff | Realistic short-range interactions with $O(N^2)$ scaling |
| Minimum image convention | Correct nearest-image distances for periodic boundaries |
| Newton's third law ($\mathbf{F}_{ij} = -\mathbf{F}_{ji}$) | Halves the number of pair evaluations |
| Verlet neighbor lists + linked-cell grid | Reduces force checks to nearby particles and enables scalable runs |
| Thread-local accumulators + post-loop reduction | Race-free parallel force summation without per-interaction synchronization |
| `guided` OpenMP schedule | Balances non-uniform per-particle workloads across threads |
| Velocity rescaling energy injection | Precisely controls energy input/output for driven simulations |
| Flat `std::array<double, d>` storage | Cache-friendly, compile-time dimensionality |
| `cnpy` NPY output | Stores large runs safely without NPZ ZIP-size limits |
| Parallel animation rendering | Multiprocessing + ffmpeg concat avoids serial matplotlib bottleneck |
| Profile-guided optimization (PGO) | Compiler uses runtime data for branch prediction and inlining |

---

## Project Structure

```
Project 8: Molicular Dynamics/
|-- main.cpp        # Entry point: reads config path and launches simulation
|-- processing.h    # MolucularSystem class: integration, forces, energy, I/O
|-- Makefile        # Multi-target build: debug, release, unsafe, profile-guided
|-- simulation_1.cfg  # Smaller exploratory config
|-- simulation_2.cfg  # Primary runtime configuration file
|-- plotting.py       # Python visualization: parallel animation renderer
|-- plot_energy.py    # Python post-processing: energy diagnostics and summaries
|-- job.sh          # SLURM batch script (default: 8 OpenMP CPUs)
|-- bin/            # Compiled executable
|-- slurm_out/      # SLURM output logs from HPC runs
`-- output/
    |-- Test Out/           # Early test run
    |-- out_0/ -- out_9/     # Simulation runs (auto-incrementing)
  |   |-- positions.npy          # Sampled positions
  |   |-- temperatures.npy       # Temperature time series
  |   |-- potentialEnergies.npy  # Potential energy time series
  |   |-- kineticEnergies.npy    # Kinetic energy time series
  |   |-- totalEnergies.npy      # Total energy time series
  |   |-- metadata.npy           # width, height, particle count, timeSteps, finalTime
  |   |-- energy_data.csv        # Energy & temperature time series (CSV)
  |   |-- positions_data.csv     # Sampled particle trajectories (CSV)
  |   |-- md_animation_N.mp4     # Particle trajectory animation
  |   |-- energy_analysis_N.png  # Composite energy diagnostics figure
  |   |-- temperature_over_energy_N.png  # Detailed T(E) trend plot
  |   `-- energy_summary_N.csv   # Scalar diagnostics summary
    `-- ...
```

---

*Nels Buhrley — Computational Physics, 2026*

<p align="center">
  <a href="https://nels.buhrley.com/CPP-Numerical-Modeling/#academic-projects"><strong>← Back to Portfolio Hub</strong></a>
</p>
