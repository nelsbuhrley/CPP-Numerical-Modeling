<p align="center">
  <a href="https://nels.buhrley.com/CPP-Numerical-Modeling/#personal-projects"><strong>← Back to Portfolio Hub</strong></a>
</p>

# Personal Project 2: Euler's Idoneal Numbers — Parallel Sieve

> [!TIP]
> Every highlighted link in this page is clickable.
> For fast navagation use [Table of Contents](#table-of-contents)


**Author:** Nels Buhrley
**Language:** C++17 with OpenMP
**Build:** `make release` — see [Build & Run](#build--run)

---

## Snapshot

- Designed and implemented a parallel number-theory sieve in C++17/OpenMP to test idoneal-number candidates up to 50,000,000.
- Optimized a triple-loop marking algorithm for large search spaces with thread-safe updates and scalable CPU utilization.
- Produced computational verification against Euler's known idoneal list and explored the frontier for undiscovered candidates.
- Demonstrates capability in parallel algorithm engineering, mathematical modeling, and large-range integer computation.

For a fast technical check, jump to [Algorithm & Parallelism](#algorithm--parallelism)

---

## Table of Contents

- [Snapshot](#snapshot)
- [Overview](#overview)
- [Technical Background](#technical-background)
  - [Euler's Idoneal Numbers](#eulers-idoneal-numbers)
  - [Known Idoneal Numbers](#known-idoneal-numbers)
  - [The Sieve Approach](#the-sieve-approach)
- [Algorithm & Parallelism](#algorithm--parallelism)
  - [OpenMP Parallelization](#openmp-parallelization)
  - [Performance](#performance)
- [Sources of Error and Limitations](#sources-of-error-and-limitations)
- [Build & Run](#build--run)
  - [Prerequisites](#prerequisites)
  - [Build](#build)
  - [Run](#run)
- [Simulation Parameters](#simulation-parameters)
- [Key Techniques](#key-techniques)
- [Project Structure](#project-structure)

---

## Overview

This project implements a **massively parallel sieve** to discover **Euler's idoneal (numeri idonei) numbers** — a rare class of integers with deep connections to quadratic forms and number theory. The sieve tests integers up to 50,000,000 using an OpenMP-parallelized triple loop, marking all values representable in a certain form and collecting the unmarked "survivors." The result is a computational verification of Euler's original list and an exploration of whether any undiscovered idoneal numbers exist beyond the known 65.

---

## Technical Background

### Euler's Idoneal Numbers

A positive integer $n$ is called **idoneal** (from the Latin *numerus idoneus*, "suitable number") if it **cannot** be written in the form:

$$n = ab + bc + ac$$

for distinct positive integers $a < b < c$.

Equivalently, $n$ is idoneal if every integer that can be represented in the form $x^2 + ny^2$ (with $\gcd(x, ny) = 1$) is necessarily a prime power, a power of 2, or twice a prime power. Euler used these numbers as a tool for proving the primality of large numbers — if $n$ is idoneal and $m = x^2 + ny^2$ has only one such representation, then $m$ is prime.

### Known Idoneal Numbers

Euler discovered 65 idoneal numbers. The complete known list is:

> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 18, 21, 22, 24, 25, 28, 30, 33, 37, 40, 42, 45, 48, 57, 58, 60, 70, 72, 78, 85, 88, 93, 102, 105, 112, 120, 130, 133, 165, 168, 177, 190, 210, 232, 240, 253, 273, 280, 312, 330, 345, 357, 385, 408, 462, 520, 760, 840, 1320, 1365, 1848

It is conjectured (and proved under the Generalized Riemann Hypothesis) that these are the **only** idoneal numbers — there are no others. Unconditionally, at most one additional idoneal number could exist, and if it does, it would be extremely large.

### The Sieve Approach

Rather than testing the quadratic-form characterization (which requires factoring), the code directly sieves on the representation $n = ab + bc + ac$:

1. **Allocate** a boolean array `isNonIdoneal[0..limit]` initialized to `false`
2. **Mark:** For all $1 \leq a < b < c$ such that $ab + bc + ac \leq \text{limit}$:
   - Compute $v = ab + bc + ac$
   - Set `isNonIdoneal[v] = true`
3. **Collect:** All indices $n$ where `isNonIdoneal[n]` remains `false` are idoneal candidates

The constraint $a < b < c$ ensures all three integers are distinct. The triple loop has complexity $\mathcal{O}(N^{3/2})$ in the limit (since $c$ is bounded by $\sim \sqrt{N/a}$ for fixed $a$), making it feasible up to tens of millions with parallelism.

---

## Algorithm & Parallelism

### OpenMP Parallelization

The outermost loop over $a$ is parallelized:

```cpp
#pragma omp parallel for schedule(dynamic)
for (long long a = 1; a * a * 3 + a * 2 <= limit; a++) {
    for (long long b = a + 1; a * b + (a + b) * (b + 1) <= limit; b++) {
        for (long long c = b + 1; a * b + b * c + a * c <= limit; c++) {
            long long val = a * b + b * c + a * c;
            isNonIdoneal[val] = true;   // atomic by design -- only writes true
        }
    }
}
```

**Why `schedule(dynamic)`:** The inner loop bounds depend on $a$ — small values of $a$ produce far more $(b, c)$ pairs than large values. Dynamic scheduling distributes these unequal chunks across threads as they become available, maintaining high utilization. Static scheduling would leave threads idle once they exhaust their smaller chunks.

**Thread safety:** The sieve array experiences only **monotonic writes** (`false` → `true`). Multiple threads may write to the same index simultaneously, but since all writes set the same value (`true`), no data race affects correctness. This is a "write-only sieve" pattern that requires no synchronization.

### Performance

The Makefile supports release builds with aggressive optimization:

```bash
-O2 -march=native -fopenmp -ftree-vectorize -ffast-math
```

The `march=native` flag enables architecture-specific SIMD instructions, and `-ftree-vectorize` allows the compiler to auto-vectorize the inner loops where possible.

---

## Sources of Error and Limitations

| Source | Nature | Mitigation |
|---|---|---|
| **Finite search limit** | Only tests up to 50,000,000; a hypothetical 66th idoneal number could be larger | Increase `limit`; GRH implies 65 is complete |
| **Memory usage** | Boolean array of size `limit` requires ~50 MB at default | Scales linearly; 500 million would need ~500 MB |
| **No primality verification** | The sieve finds candidates but does not verify the quadratic-form property | Cross-reference with Euler's published list |
| **Integer overflow** | Product $ab + bc + ac$ can overflow for very large limits | `long long` (64-bit) safe up to ~$3 \times 10^{18}$ |

---

## Build & Run

### Prerequisites

- **C++17** compatible compiler with OpenMP support (`g++` or `clang++`)
- On macOS: `brew install libomp` for OpenMP with Clang

### Build

```bash
# Optimized release build
make release

# Default build
make

# Clean build artifacts
make clean
```

The Makefile auto-detects the compiler. On the BYU Marylou supercomputer, it uses `g++` with GCC-native OpenMP.

### Run

```bash
./my_program
```

**Output (console):**
```
Number of idoneal numbers found below 50000000: 65
Idoneal numbers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 18, 21, ...
Time taken: X.XX seconds
```

---

## Simulation Parameters

Configured in [main.cpp](main.cpp):

| Parameter | Default | Description |
|---|---|---|
| `limit` | 50,000,000 | Upper bound of the sieve search |
| OpenMP threads | Auto (system max) | Set via `OMP_NUM_THREADS` environment variable |

---

## Key Techniques

| Technique | Purpose |
|---|---|
| Triple-loop sieve over $a < b < c$ | Direct computation of all representable values $ab + bc + ac$ |
| Monotonic write-only sieve pattern | Thread-safe without locks — all writes are idempotent |
| `schedule(dynamic)` on outer loop | Balances highly unequal work across threads |
| `long long` arithmetic | Prevents overflow for products up to $\sim 10^{18}$ |
| Tight loop bounds from algebraic constraints | Minimizes unnecessary iterations; $c$ bounded by $\sqrt{N/a}$ |
| `-march=native -ftree-vectorize` | Architecture-specific SIMD and auto-vectorization |

---

## Project Structure

```
Personal Project 2: Idelic Numbers/
|-- main.cpp      # Sieve implementation with OpenMP parallelization
|-- Makefile      # Build targets: default, release, clean
|-- my_program    # Compiled executable
`-- bin/
    `-- main      # Alternative executable location
```

---

*Nels Buhrley — Computational Mathematics, 2025*

<p align="center">
  <a href="https://nels.buhrley.com/CPP-Numerical-Modeling/#personal-projects"><strong>← Back to Portfolio Hub</strong></a>
</p>
