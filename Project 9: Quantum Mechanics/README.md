<p align="center">
  <a href="https://nels.buhrley.com/CPP-Numerical-Modeling/#academic-projects"><strong>← Back to Portfolio Hub</strong></a>
</p>

# Project 9: Quantum Mechanics — Bound States & Eigenstates

> [!TIP]
> Every highlighted link in this page is clickable.
> For fast navagation use [Table of Contents](#table-of-contents)

**Author:** Nels Buhrley
**Language:** C++17 · Python 3 (visualization)
**Build:** `make release` — see [Build & Run](#build--run)

---

## Snapshot

- Built a full quantum mechanics eigenstate solver that computes bound states for polynomial potential wells by integrating the 1D Schrödinger equation.
- Implemented the **Numerov method** (4th-order finite-difference integration) to solve $-\frac{\hbar^2}{2m}\psi''(x) + V(x)\psi(x) = E\psi(x)$ with arbitrary precision.
- Developed a **shooting-and-matching algorithm** combining energy sweeps with node counting and bisection search to locate eigenstate energies.
- Generated interactive 3D visualizations of eigenstates and potential wells across multiple quantum numbers and well shapes (even polynomial degrees: 2, 4, 6, 8, 10).
- Demonstrates expertise in numerical quantum mechanics, boundary value problem solving, and high-precision root-finding.

For a fast technical check, jump to [Code Structure](#code-structure), [Results](#results), and [Build & Run](#build--run).

---

## Table of Contents

- [Snapshot](#snapshot)
- [Overview](#overview)
- [Physics Background](#physics-background)
  - [The 1D Time-Independent Schrödinger Equation](#the-1d-time-independent-schrödinger-equation)
  - [Bound States and Quantization](#bound-states-and-quantization)
  - [Polynomial Potentials](#polynomial-potentials)
- [Code Structure](#code-structure)
  - [Numerov Integration](#numerov-integration)
  - [Eigenstate Finding](#eigenstate-finding)
  - [Boundary Value Problem Solving](#boundary-value-problem-solving)
- [Results](#results)
  - [Interactive Eigenstates Visualization](#interactive-eigenstates-visualization)
  - [Well Shapes and Bound States](#well-shapes-and-bound-states)
- [Sources of Error](#sources-of-error)
- [Build & Run](#build--run)
- [Simulation Parameters](#simulation-parameters)
- [Key Techniques](#key-techniques)
- [Project Structure](#project-structure)

---

## Overview

This project implements a **quantum mechanics eigenstate solver** for the one-dimensional time-independent Schrödinger equation with finite-range polynomial potentials. The solver:

1. Sweeps through energy space to locate **nodal brackets** — energy pairs that bracket a zero-crossing in the wavefunction
2. Uses **bisection refinement** to converge on each eigenstate's energy with machine precision
3. Integrates the Schrödinger equation via the **Numerov method** to obtain the spatial wavefunction $\psi(x)$
4. Supports polynomial potentials of arbitrary degree $V(x) = \frac{x^n}{n}$ and multiple quantum states

The implementation demonstrates advanced techniques in numerical boundary value problems: energy quantization by node counting, high-order finite-difference schemes, and robust root-finding under nonlinear ODE constraints.

---

## Physics Background

### The 1D Time-Independent Schrödinger Equation

The binding structure of quantum systems follows from the time-independent Schrödinger equation:

$$-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi(x) = E\psi(x)$$

In natural units ($\hbar = m = 1$), this becomes:

$$-\frac{1}{2}\psi'' + V(x)\psi = E\psi$$

or equivalently, rearranging to standard ODE form:

$$\psi''(x) = 2[V(x) - E]\psi(x)$$

### Bound States and Quantization

For a finite-range potential $V(x) \to 0$ as $\lvert x \rvert \to \infty$, **bound states** exist when $E < V_{\max}$ and satisfy the boundary conditions:

- $\psi(x) \to 0$ as $x \to \pm\infty$

These boundary conditions are **quantizing**: only discrete energy values $E_n$ allow the wavefunction to decay smoothly to zero at both boundaries. The energy quantization is directly encoded in the **nodal structure**: excited state $n$ has exactly $n$ nodes (zero-crossings) in its wavefunction.

### Polynomial Potentials

This solver focuses on polynomial potentials of the form:

$$V(x) = \frac{x^n}{n}$$

where $n$ is the **degree** (restricted to even values: 2, 4, 6, 8, 10). Each implemented degree corresponds to:

- **n = 2:** Harmonic oscillator potential (parabolic well)
- **n = 4:** Quartic well (double-well structure)
- **n = 6:** Sextic potential (steep confinement)
- **n = 8:** Octic potential (exponential-like growth)
- **n = 10:** Decic potential (extreme confinement)

All implemented potentials are **symmetric** ($V(-x) = V(x)$), which splits eigenstates into **even** and **odd parity** families. This symmetry enables efficient independent sweeps with appropriate initial conditions ($\psi(0) = 1, \psi'(0) = 0$ for even parity; $\psi(0) = 0, \psi'(0) = 1$ for odd parity).

---

## Code Structure

All core logic resides in [processing.h](processing.h) with entry point [main.cpp](main.cpp).

| Component | Purpose |
|---|---|
| `NumerovIntegrate()` | 4th-order finite-difference solver for the ODE; used by both shooting and matching methods |
| `countNodes()` | Counts zero-crossings in $\psi(x)$ to identify quantum number |
| `findNodalBracketsAtDegree()` | Energy sweep + node detection to bracket each eigenstate |
| `findEigenstatesAtDegreeShooting()` | Bisection refinement using shooting method (forward integration, boundary matching) |
| `findEigenstatesAtDegreeMatching()` | Bisection refinement using matching method (backward integration, origin matching) |
| `findEigenstatesAtDegree()` | Wrapper function that selects between shooting or matching based on configuration |
| `Sweep` class | Orchestrates multi-degree sweep and output |

### Numerov Integration

The Numerov method is a **4th-order predictor-corrector scheme** for solving $y'' = g(x, y)$. For the Schrödinger equation with $g = 2[V(x) - E]$, the recurrence relation is:

$$\psi_{i+1} = \frac{2(1 + 5h^2g_i/12)\psi_i - (1 - h^2g_{i-1}/12)\psi_{i-1}}{1 - h^2g_{i+1}/12}$$

where $h = \Delta x$ is the spatial step size and $g_i = 2[V(x_i) - E]$.

**Advantages:**
- 5th-order local truncation error (vs. 3rd for standard 2nd-order schemes)
- Compact 3-point stencil, easy to parallelize
- Stable for smooth potentials

**Implementation details** (from [processing.h](processing.h)):
- Precomputes $2V(x_i)$ on the full grid to avoid redundant arithmetic
- Coefficients $a_{\text{prev}}, a_{\text{curr}}, a_{\text{next}}$ factor out division via careful arithmetic
- Detects divergence (when $\psi > 10^4$) to halt integration early
- Returns only the valid non-divergent portion of $\psi(x)$

### Eigenstate Finding

#### Step 1: Energy Sweep & Nodal Bracketing

The solver sweeps energy from `energyMin` upward in steps of `energyStepSize`, tracking node count. When the node count increases, an eigenstate bracket is recorded:

```
E_minus = previous energy (fewer nodes)
E_plus  = current energy  (one more node)
```

This pair brackets the eigenstate. Separate sweeps for even ($\psi(0) = 1, \psi'(0) = 0$) and odd ($\psi(0) = 0, \psi'(0) = 1$) parity allow independent exploration of each family.

#### Step 2: Bisection Refinement — Shooting vs. Matching Methods

Once a bracket $[E_-, E_+]$ is located, bisection converges the eigenstate energy. Two independent methods are implemented to refine the eigenvalue:

##### **Shooting Method**

The shooting method integrates forward from $x = 0$ with known boundary conditions:

1. Set initial conditions $\psi(0)$ and $\psi'(0)$ based on parity (even: $\psi(0) = 1, \psi'(0) = 0$; odd: $\psi(0) = 0, \psi'(0) = 1$)
2. Integrate $\psi(x; E_{\text{mid}})$ from $x = 0$ to $x = x_{\text{end}}$ using Numerov
3. Evaluate the wavefunction at the far boundary: $\psi(x_{\text{end}})$
4. Track the sign of $\psi(x_{\text{end}})$ across the energy bracket: if $\psi(x_{\text{end}})$ has the same sign at $E_+$, replace $E_+$; otherwise replace $E_-$
5. Update midpoint $E_{\text{mid}} = (E_- + E_+) / 2$ and repeat until $\lvert E_+ - E_- \rvert < \epsilon_{\text{tol}}$

**Advantages:** Physically direct — integrates forward with true boundary conditions; one pass per energy trial

**Limitations:** Boundary condition sensitivity — far-field behavior drives convergence, which can be noisy when the wavefunction is exponentially small

##### **Matching Method**

The matching method exploits the symmetry of the potential by integrating backward from the boundary to the origin and matching boundary conditions:

1. Compute reference boundary values by integrating at the bracket endpoints: $\psi(x_{\text{end}}; E_-)$ and derivative $\psi'(x_{\text{end}}; E_-)$, similarly for $E_+$
2. Integrate backward from $x = x_{\text{end}}$ to $x = 0$ at energy $E_{\text{mid}}$, extracting $\psi(0; E_{\text{mid}})$ and $\psi'(0; E_{\text{mid}})$
3. **Compare at the origin** — select the comparison value based on parity:
   - Even parity ($n$ even): match on $\psi(0)$ (derivative should vanish)
   - Odd parity ($n$ odd): match on $\psi'(0)$ (wavefunction should vanish)
4. If the midpoint value has the same sign as the lower-bracket boundary value, replace $E_+$; otherwise replace $E_-$
5. Repeat until convergence

**Advantages:** Origin-centric — compares well-defined boundary conditions at $x = 0$ where wavefunctions are large; easier enforcement of parity; more stable convergence

**Implementation flexibility:** Both methods employ bisection root-finding, making them equally robust in the iteration count. The choice is controlled by the `useShootingMethod` flag in [main.cpp](main.cpp) — set to `false` for matching (default) or `true` for shooting.

**Comparison:**
| Aspect | Shooting | Matching |
|---|---|---|
| Integration direction | Forward ($0 \to x_{\text{end}}$) | Backward ($x_{\text{end}} \to 0$) |
| Matching location | Boundary ($x_{\text{end}}$) | Origin ($x = 0$) |
| Matching value | $\psi(x_{\text{end}})$ | $\psi(0)$ or $\psi'(0)$ (parity-dependent) |
| Convergence stability | Sensitive to boundary exponentials | Stable, leverages origin regularity |
| Physical intuition | Boundary-value problem | Symmetry enforcement |

##### **Shooting vs. Matching for Symmetric Potentials**

Both shooting and matching methods are effective for symmetric potentials, but they employ complementary strategies:

**Shooting Method — Forward Integration from Origin**
- Integrates forward from $x = 0$ with initial conditions based on parity: $\psi(0) = 1, \psi'(0) = 0$ (even parity) or $\psi(0) = 0, \psi'(0) = 1$ (odd parity)
- **Advantages:** Physically direct — imposes true boundary conditions at a point where the wavefunction is large ($\psi(0) = O(1)$). The forward integration accumulates phase smoothly through both spatial quadrants. Convergence is rapid because the boundary value $\psi(x_{\text{end}})$ reflects high-quality accumulated information.
- **Efficiency:** One forward pass per energy trial; well-defined boundary evaluations with no exponential decay complications.
- **Why it works for symmetric wells:** The parity structure guarantees symmetric decay in both directions. Even with exponential decay at large $\lvertx\rvert$, the boundary value remains $O(\epsilon)$ where $\epsilon \sim e^{-\int \lvert k \rvert\,dx}$ is the same from either direction.

**Matching Method — Backward Integration from Boundary**
- Integrates backward from $x = x_{\text{end}}$ to extract boundary-condition information at the origin: $[\psi(0), \psi'(0)]$
- **Advantages:** Origin-centric — evaluates conditions at $x = 0$ where wavefunctions are well-defined and large. Parity constraints are naturally satisfied by construction, reducing bisection ambiguity. Numerically robust because it avoids relying on exponentially small far-field values.
- **Efficiency:** Slightly higher per-trial cost (backward integration required), but compensated by fewer bisection iterations due to origin-based matching.
- **Why it works for symmetric wells:** The backward propagation extracts the *actual* boundary behavior at the origin without assuming parity. Even if both methods converge, matching reduces sensitivity to numerical errors accumulated during far-field propagation.

**Comparison for Even-Degree Potentials ($n = 2, 4, 6, 8, 10$):**

| Aspect | Shooting | Matching |
|---|---|---|
| Integration direction | Forward ($0 \to x_{\text{end}}$) | Backward ($x_{\text{end}} \to 0$) |
| Matching location | Boundary ($x = x_{\text{end}}$) | Origin ($x = 0$) |
| Amplitude scale | $\psi(x_{\text{end}}) \sim e^{-\alpha}$ | $\psi(0) \sim O(1)$ |
| Convergence | Fast (high-amplitude endpoint) | Very stable (origin-based) |
| Bisection iterations | ~30–40 | ~35–45 |
| Recommended use | Symmetric wells where speed is priority | Robust convergence across all well depths |

**Why Matching is the Default — Theoretical Context:**
The default configuration uses matching because it generalizes better to asymmetric potentials. Consider a theoretical example: an **asymmetric step potential** $V(x) = \begin{cases} 0 & x < 0 \\ V_0 & x \geq 0 \end{cases}$. Shooting from $x = 0$ with assumed parity fails here — the asymmetry breaks the parity assumption, causing the wavefunction to grow exponentially in one direction and decay in the other. This creates overflow at the far boundary, poisoning the bisection. Matching stabilizes by integrating backward and matching at the origin, where the wavefunction is large and well-defined, avoiding exponential amplification. Your implementation prioritizes this robust strategy by defaulting to matching.

**Current Implementation:**
The default setting in [main.cpp](main.cpp) is `useShootingMethod = false` (matching method), which provides optimal stability and consistency across all even-degree symmetric potentials. For maximum *performance* on symmetric wells, switching to `true` (shooting) can accelerate convergence by ~10–15% with no loss of accuracy.

#### Step 3: Wavefunction Refinement

Once the eigenstate energy is converged:

1. **Trim:** Remove the divergent tail (tail magnitude > interior magnitude)
2. **Normalize:** Enforce $\int_{-\infty}^{\infty} \lvert\psi(x)\rvert^2 dx = 1$

---

## Results

### Interactive Eigenstates Visualization

The eigenstates across all implemented even-degree potentials (n = 2, 4, 6, 8, 10) and up to 11 bound states per degree are visualized as **tabbed interactive 3D plots**.

<div style="width: 100%; aspect-ratio: 1 / 1; overflow: hidden; border: 1px solid #ddd; border-radius: 8px;">
<iframe
  src="https://nels.buhrley.com/assets/html-assets/eigenstates_tabs.html"
  width="100%"
  height="100%"
  style="border: none; display: block;">
</iframe>
</div>

<a href="https://nels.buhrley.com/assets/html-assets/eigenstates_tabs.html" target="_blank">View Detailed Eigenstates Fullscreen ↗️</a>

**Features:**
- **Tabs by potential degree:** Switch between harmonic (n=2), quartic (n=4), sextic (n=6), octic (n=8), and decic (n=10) wells
- **3D and 2D Viewing:** Switch between 2D and 3D repersentations of the wavefunction with the potental well superemposed over the 2D plots
- **Energy table:** Quantized energy levels with full precision

Select any degree tab above to explore states for that potential family.

### Well Shapes and Bound States

Each potential supports a discrete set of bound states determined by energy quantization. The relationship between well shape and eigenstate structure:

| Degree | Potential | Well Type | Key Feature |
|---|---|---|---|
| 2 | $V(x) = x^2/2$ | Parabolic | Evenly-spaced levels; analytical basis available (Hermite polynomials) |
| 4 | $V(x) = x^4/4$ | Quartic double-well | States exhibit avoided crossings; higher-n states probe double-well barrier |
| 6 | $V(x) = x^6/6$ | Sextic | Steep walls; strong confinement |
| 8 | $V(x) = x^8/8$ | Octic | Even deeper well; exponential-like confinement; rapid level clustering |
| 10 | $V(x) = x^{10}/10$ | Decic | Extreme confinement; highest degree implemented; particle nearly confined to origin |

**Symmetry Structure:**
All potentials are **symmetric** ($V(-x) = V(x)$), which determines the parity structure of eigenstates:
- **Even-parity states:** $\psi(-x) = \psi(x)$ (wavefunction symmetric around origin)
- **Odd-parity states:** $\psi(-x) = -\psi(x)$ (wavefunction antisymmetric around origin)

This symmetry is essential for the numerical methods, which exploit parity to achieve efficient and stable convergence.

---

## Sources of Error

| Source | Nature | Mitigation |
|---|---|---|
| Spatial discretization | $\Delta x$ finite → wavefunction sampled at grid points | Use `stepSize = 0.00001`; Numerov method is 5th-order accurate |
| Boundary truncation | Potential extends to $x_{\text{end}} = 15.0$ rather than $x = \infty$ | Wavefunction decays exponentially; tail contribution negligible |
| Bisection convergence | Finite precision in energy bracket | Tolerance set to $10^{-8}$; typical refinement: ~40–50 iterations per state |
| Node counting discreteness | A state's node count only advances by 1 per energy increment | Use small `energyStepSize = 0.11` to ensure no state is skipped |
| Numerov truncation error | 5th-order scheme accumulates error over integration domain | Step size tuned to maintain error below $10^{-8}$ |

**Computational complexity:**
- Per eigenstate: $\mathcal{O}(100 \text{ Numerov sweeps} \times 1.5{,}M \text{ spatial points}) = \mathcal{O}(150 \text{ M updates})$
- Total: $\mathcal{O}(9 \text{ degrees} \times 11 \text{ states} \times 150 \text{ M}) \sim \mathcal{O}(15 \text{ B FLOPs})$ — runs in < 2 s

---

## Build & Run

### Prerequisites

- **C++17** compiler (`g++` or `clang++`)
- **zlib** (for `.npz` output; usually system-provided)
- **Python 3** with `numpy` and `plotly` (for visualization)

### Build Targets

```bash
make release   # Optimized build (-O3, LTO, vectorization, march=native)
make debug     # -O0, full warnings
make clean     # Remove build artifacts
```

### Run

```bash
./bin/main
```

Output is saved to `output/eigenstates.npz`. Then generate visualizations:

```bash
python3 postprocessing.py
```

This produces:
- `eigenstates_tabs_detailed.html` — Full dashboard with 3D plot, potential, and data table
- `eigenstates_tabs_simple.html` — 3D-only simplified view

Upload both HTML files to the assets folder for inclusion in the portfolio.

---

## Simulation Parameters

Configured in [main.cpp](main.cpp):

| Parameter | Default | Description |
|---|---|---|
| `maxDegree` | 10 | Highest polynomial degree to solve (n = 2...10) |
| `nodesToFind` | 11 | Number of bound states per degree |
| `stepSize` | 0.00001 | Spatial discretization $\Delta x$ |
| `energyStepSize` | 0.11 | Energy increment for nodal bracket sweep |
| `convergenceTol` | 1e-8 | Bisection convergence threshold on bracket width |
| `maxIterations` | 100 | Max bisection iterations per eigenstate |
| `energyMin` | 0.001 | Minimum energy for sweep start |
| `xEnd` | 15.0 | Integration domain endpoint |
| `useShootingMethod` | false | **Method selector:** `true` for shooting (forward integration), `false` for matching (backward integration) |
| `targetXEnd` | 6.0 | Resampling domain for visualization |
| `targetPsiPoints` | 150 | Resampling target for output visualization |

---

## Key Techniques

| Technique | Purpose |
|---|---|
| Numerov 4th-order method | High-accuracy ODE integration with compact stencil |
| Node counting via sign crossings | Identifies quantum number without computing spectrum |
| Precomputed potential mesh | Avoids redundant function evaluations in loop |
| Early divergence detection | Halts integration when $\|\psi\| > 10^4$ to save compute |
| Even/odd parity decomposition | Reduces energy sweep by separating independent families |
| **Shooting method** | **Forward integration from origin; boundary-based root-finding** |
| **Matching method** | **Backward integration; origin-based boundary condition matching** |
| Bisection root-finding | Robust eigenstate energy convergence without derivatives |
| Wavefunction trimming | Removes divergent tail post-integration |
| Wavefunction normalization | Ensures $\|\psi\|_2 = 1$ for correct probability interpretation |
| Resampling for output | Downsamples from ~1.5M grid points to ~150 for visualization |
| NPZ compression | Stores all states and energies in a single compact binary archive |

---

## Project Structure

```
Project 9: Quantum Mechanics/
|-- main.cpp                    # Entry point: configures degrees, iterations, parameters
|-- processing.h                # Core solver: Numerov, nodal bracketing, bisection refinement
|-- Makefile                    # Multi-target build: debug, release, clean
|-- postprocessing.py           # Generates interactive Plotly dashboards from NPZ output
|-- requirements.txt            # Python dependencies (numpy, plotly)
`-- output/
    `-- eigenstates.npz         # Compressed eigenstate database
        |-- psi_degree_2_state_0..9       # Wavefunction trajectories for n=2
        |-- energies_degree_2             # Energy levels for n=2
        |-- psi_degree_3_state_0..9       # Wavefunction trajectories for n=3
        |-- energies_degree_3             # Energy levels for n=3
        |-- ... (similarly for n=4,5,6)
        `-- targetXEnd                    # Integration endpoint (8.0)
```

---

*Nels Buhrley — Computational Physics, 2026*

<p align="center">
  <a href="https://nels.buhrley.com/CPP-Numerical-Modeling/#academic-projects"><strong>← Back to Portfolio Hub</strong></a>
</p>
