<p align="center">
  <a href="https://nels.buhrley.com/CPP-Numerical-Modeling/#personal-projects"><strong>← Back to Portfolio Hub</strong></a>
</p>

# Personal Project 1: The Collatz Conjecture (3n+1)

> [!TIP]
> Every highlighted link in this page is clickable.
> For fast navagation use [Table of Contents](#table-of-contents)


**Author:** Nels Buhrley
**Language:** C++17 · Python 3 (visualization)

---

## Snapshot

- Built a high-throughput C++ pipeline to analyze Collatz dynamics over very large integer ranges and extract stopping-time and peak-value statistics.
- Implemented multiple tools for different workloads: exhaustive batch analysis, interactive sequence exploration, and CSV data generation for downstream analytics.
- Produced publication-ready Python visualizations that reveal nontrivial structure in stopping times and maximum excursions.
- Demonstrates strong fundamentals in algorithm design, performance-minded implementation, and data-driven mathematical investigation.

For a fast technical check, jump to [Programs](#programs) and [Building and Running](#building-and-running).

---

## Table of Contents

- [Snapshot](#snapshot)
- [Overview](#overview)
- [Technical Background](#technical-background)
  - [The Collatz Conjecture](#the-collatz-conjecture)
  - [Key Quantities](#key-quantities)
  - [Growth and Decay](#growth-and-decay)
- [Programs](#programs)
  - [`3n+1.cpp` — Exhaustive Stopping-Time Survey](#3n1cpp--exhaustive-stopping-time-survey)
  - [`3n+1#2.cpp` — Interactive Sequence Explorer](#3n12cpp--interactive-sequence-explorer)
  - [`collatz_data_generator.cpp` — CSV Data Generation](#collatz_data_generatorcpp--csv-data-generation)
  - [`plot_collatz.py` — Visualization](#plot_collatzpy--visualization)
- [Building and Running](#building-and-running)
  - [Prerequisites](#prerequisites)
  - [Compile](#compile)
  - [Run](#run)
- [Sources of Error and Limitations](#sources-of-error-and-limitations)
- [Key Techniques](#key-techniques)
- [Project Structure](#project-structure)

---

## Overview

This project is a computational exploration of the **Collatz conjecture** — one of the most famous unsolved problems in mathematics. Three independent C++ programs investigate different aspects of the conjecture by computing Collatz sequences for millions of starting values, tracking stopping times, finding extremal elements, and generating data for visual analysis. A Python script plots the results, revealing the fractal-like structure hidden in this deceptively simple iteration.

---

## Technical Background

### The Collatz Conjecture

For any positive integer $n$, define the function:

$$T(n) = \begin{cases} n/2 & \text{if } n \text{ is even} \\ 3n + 1 & \text{if } n \text{ is odd} \end{cases}$$

The **Collatz conjecture** (also known as the $3n+1$ problem, Ulam's conjecture, or the Syracuse problem) states that for every positive integer $n$, the iterated sequence $n, T(n), T^2(n), T^3(n), \ldots$ eventually reaches 1.

Despite its elementary statement, the conjecture has resisted proof since Lothar Collatz first posed it in 1937. Paul Erdős remarked: *"Mathematics may not be ready for such problems."* It has been verified computationally for all $n < 2^{68}$ (as of 2020), but no general proof exists.

### Key Quantities

| Quantity | Definition |
|---|---|
| **Stopping time** $\sigma(n)$ | The number of iterations until the sequence first reaches a value $< n$ |
| **Total stopping time** $\sigma_\infty(n)$ | The number of iterations until the sequence reaches 1 |
| **Maximum excursion** $\max_k T^k(n)$ | The largest value visited during the sequence |

The stopping time and maximum excursion both exhibit highly irregular, seemingly random behavior as a function of $n$ — yet the conjecture asserts that $\sigma_\infty(n)$ is always finite.

### Growth and Decay

On average, even steps halve the value and odd steps multiply by ~3/2 (since $3n+1$ is followed by at least one even step). Since $\log_2(3/2) \approx 0.585$, and roughly half of all integers are even, the expected net effect per iteration is a reduction — which heuristically "explains" why sequences tend to collapse toward 1, though this argument falls short of a proof.

---

## Programs

### `3n+1.cpp` — Exhaustive Stopping-Time Survey

Computes the Collatz total stopping time for every integer from 1 to 1,000,000. Identifies the starting value that requires the most iterations to reach 1.

**Algorithm:**
```
for n = 1 to 1,000,000:
    count = 0
    x = n
    while x != 1:
        x = T(x)
        count++
    if count > max_count:
        max_count = count
        max_start = n
```

**Output:** The starting value with the longest total stopping time and its iteration count.

### `3n+1#2.cpp` — Interactive Sequence Explorer

An interactive program that follows a recursive pattern: starting at $n = 1$, it runs the Collatz sequence, finds the maximum value $M$ reached, then restarts at $M + 1$. This traces a "ladder" through the integer line, visiting progressively larger starting values that reach progressively higher peaks.

The user presses **SPACE** to continue to the next cycle or **q** to quit, making it a hands-on tool for developing intuition about Collatz sequence behavior.

### `collatz_data_generator.cpp` — CSV Data Generation

Generates a CSV file (`collatz_data.csv`) recording, for each iteration of the $(3n+1)+1$ recursive process, the maximum value reached. This data is consumed by the Python visualization script.

**Output format:**
```csv
index,max_value
0,2
1,16
2,160
...
```

### `plot_collatz.py` — Visualization

Reads `collatz_data.csv` and plots the maximum value reached vs. iteration index. Automatically switches to logarithmic scale when the range spans multiple orders of magnitude. Prints summary statistics (min, max, mean, median) to the console.

Plots can optionally be **saved to disk** in any format supported by Matplotlib (PNG, PDF, SVG, …) via the `-o` / `--output` flag. The `--no-show` flag suppresses the interactive window, which is useful in headless or CI environments.

---

## Building and Running

### Prerequisites

- **C++17** compatible compiler (`g++` or `clang++`)
- **Python 3** with `pandas` and `matplotlib`

### Compile

```bash
# Stopping-time survey
g++ -std=c++17 -O2 3n+1.cpp -o collatz_survey

# Interactive explorer
g++ -std=c++17 -O2 "3n+1#2.cpp" -o collatz_explorer

# Data generator
g++ -std=c++17 -O2 collatz_data_generator.cpp -o collatz_data_gen
```

### Run

```bash
# Full survey (1--1,000,000)
./collatz_survey

# Interactive mode
./collatz_explorer

# Generate CSV data for plotting
./collatz_data_gen

# Visualize (interactive)
python3 plot_collatz.py

# Visualize and save plot to PNG
python3 plot_collatz.py -o collatz_plot.png

# Save to PNG without opening an interactive window (headless / CI)
python3 plot_collatz.py collatz_data.csv -o collatz_plot.png --no-show

# Save as PDF
python3 plot_collatz.py -o collatz_plot.pdf
```

---

## Sources of Error and Limitations

| Source | Nature | Mitigation |
|---|---|---|
| **Integer overflow** | Collatz sequences can temporarily grow very large (e.g., $n = 113383$ reaches $2.5 \times 10^9$) | Use `long long` (64-bit) arithmetic; known safe for $n < 2^{62}$ |
| **Finite search range** | Only testing $n \leq 10^6$; conjecture could fail at larger $n$ | Computational verification has been extended to $n < 2^{68}$ by other researchers |
| **No formal proof** | Empirical verification ≠ proof | This is a computational exploration, not a proof attempt |
| **Stopping-time statistics** | The distribution of $\sigma_\infty(n)$ is conjectured to be roughly Gaussian in $\log n$ but not proven | Plotting reveals the structure; formal analysis remains open |

---

## Key Techniques

| Technique | Purpose |
|---|---|
| Brute-force iteration to $10^6$ | Exhaustive verification of the conjecture up to a practical limit |
| Recursive max-value chaining | Explores the "growth frontier" of Collatz sequences |
| CSV pipeline to Python | Separation of computation (C++) and visualization (Python) |
| Automatic log-scale detection | Handles the extreme dynamic range of max values |

---

## Project Structure

```
Personal Project 1: 3n+1/
|-- 3n+1.cpp                  # Exhaustive stopping-time survey (1--10^6)
|-- 3n+1#2.cpp                # Interactive recursive sequence explorer
|-- collatz_data_generator.cpp # CSV data generator for plotting
|-- plot_collatz.py            # Python visualization (max values vs. iteration)
`-- collatz_data.csv           # Generated data file
```

---

*Nels Buhrley — Computational Mathematics, 2025*

<p align="center">
  <a href="https://nels.buhrley.com/CPP-Numerical-Modeling/#personal-projects"><strong>← Back to Portfolio Hub</strong></a>
</p>
