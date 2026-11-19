<p align="center">
  <a href="https://nels.buhrley.com/CPP-Numerical-Modeling/#personal-projects"><strong>← Back to Portfolio Hub</strong></a>
</p>

# Personal Project 3: The Monty Hall Problem — A Probabilistic Paradox

> [!TIP]
> Every highlighted link in this page is clickable.
> For fast navagation use [Table of Contents](#table-of-contents)


**Author:** Nels Buhrley
**Language:** Python 3
**Visualization:** Matplotlib

---

## Snapshot

- Built a Monte Carlo simulation in Python to validate a counterintuitive probability result with empirical convergence to theory.
- Implemented side-by-side strategy evaluation (stay vs switch) and tracked win-rate behavior over repeated trials.
- Delivered clear statistical visualizations that communicate uncertainty reduction and asymptotic convergence.
- Demonstrates practical statistical computing, experiment design, and technical communication for non-intuitive results.

For a fast technical check, jump to [Simulation Design](#simulation-design) and [Results & Visualization](#results--visualization).

---

## Table of Contents

- [Snapshot](#snapshot)
- [Overview](#overview)
- [The Monty Hall Problem](#the-monty-hall-problem)
  - [Problem Statement](#problem-statement)
  - [The Paradox](#the-paradox)
  - [Mathematical Analysis](#mathematical-analysis)
- [Simulation Design](#simulation-design)
  - [Algorithm](#algorithm)
  - [Strategies](#strategies)
- [Results & Visualization](#results--visualization)
  - [Convergence to Theoretical Probabilities](#convergence-to-theoretical-probabilities)
  - [Final Win Rate Comparison](#final-win-rate-comparison)
- [Running the Simulation](#running-the-simulation)
  - [Prerequisites](#prerequisites)
  - [Execution](#execution)
- [Key Insights](#key-insights)
- [Project Structure](#project-structure)

---

## Overview

The **Monty Hall problem** is one of the most famous probability puzzles in mathematics, notorious for contradicting human intuition. This project implements a Monte Carlo simulation to empirically verify the surprising theoretical predictions. Rather than accepting the mathematical result at face value, I built this simulation to *visualize the probabilities converging* to their theoretical limits — watching 1000 trials reveal the hidden structure of this paradoxical game.

The simulation compares two competing strategies:
- **Stay:** Keep your initial choice after Monty reveals a goat
- **Switch:** Change your choice to the remaining unopened door

The counterintuitive result: switching doubles your probability of winning from 1/3 to 2/3, yet most people instinctively stay.

---

## The Monty Hall Problem

### Problem Statement

You are a contestant on a game show. Behind three doors are:
- **One prize** (a car)
- **Two non-prizes** (goats)

You choose a door. Monty Hall, the host, opens one of the remaining doors to reveal a goat. You now face a choice:

> **Do you stick with your original choice, or switch to the only other unopened door?**

The question asks: *which strategy maximizes your probability of winning the car?*

### The Paradox

The problem gained fame in 1990 when Marilyn vos Savant published her solution in *Parade Magazine*, claiming that switching is the winning strategy with probability 2/3, while staying wins with probability 1/3. Her column received thousands of letters from readers (including mathematicians and PhDs) insisting she was wrong.

Yet she was right. The paradox arises because human intuition applies flawed probabilistic reasoning — many people believe that after Monty opens a door, the remaining two doors are equally likely to contain the prize (probability 1/2 each). This "symmetry intuition" is **incorrect**.

### Mathematical Analysis

#### Staying Strategy

Your initial choice has a $\frac{1}{3}$ probability of being correct. Monty cannot change this fundamental fact: he only reveals information about the doors *you didn't choose*. Therefore:

$$P(\text{win} \mid \text{stay}) = \frac{1}{3}$$

#### Switching Strategy

Consider the initial probability partition:
- **Your door contains the car:** probability $\frac{1}{3}$
  - Monty reveals a goat from the other two doors
  - Switching gives you a goat — **you lose**

- **Your door contains a goat:** probability $\frac{2}{3}$
  - Monty *must* reveal the other goat (he cannot reveal the car)
  - Switching gives you the car — **you win**

By the law of total probability:

$$P(\text{win} \mid \text{switch}) = P(\text{initially chose goat}) = \frac{2}{3}$$

The key insight: Monty's action is *not random* — it's constrained by the locations of both the car and goats. This constraint reveals information that shifts probability from your door to the remaining door.

---

## Simulation Design

### Algorithm

For each trial:

1. **Setup:** Randomly place the car behind one of three doors (0, 1, or 2)
2. **Initial Choice:** Randomly select your initial door choice
3. **Monty's Reveal:** Monty opens a door that is:
   - Not your chosen door
   - Not the car door
   - (This determines Monty's choice uniquely when your initial choice contains a goat, but leaves one option when you initially chose the car)
4. **Stay Strategy:** Win if your initial choice equals the car door
5. **Switch Strategy:** Win if the remaining unopened door (not yours, not Monty's) contains the car
6. **Record:** Track cumulative win rates for both strategies

The simulation runs 1000 trials, computing the empirical win rate after each trial to visualize convergence to the theoretical limits.

### Strategies

- **Stay:** You maintain your initial choice regardless of which door Monty opens
- **Switch:** You always change to the one remaining unopened door after Monty's reveal

---

## Results & Visualization

### Convergence to Theoretical Probabilities

![Monty Hall Convergence Plot](monty_hall_convergence.png)

This plot shows the empirical win rate for both strategies across 1000 trials:

- **Red line (Stay):** Oscillates around the theoretical 1/3 (≈ 0.333), converging from above and below
- **Green line (Switch):** Oscillates around the theoretical 2/3 (≈ 0.667), similarly converging
- **Black dashed line:** Theoretical $P(\text{win} \mid \text{stay}) = 1/3$
- **Black dash-dot line:** Theoretical $P(\text{win} \mid \text{switch}) = 2/3$

The oscillations gradually dampen as the law of large numbers drives both empirical rates toward their theoretical values. By 1000 trials, both strategies have converged visibly close to prediction.

### Final Win Rate Comparison

![Monty Hall Comparison Bar Chart](monty_hall_comparison.png)

After 1000 trials:
- **Stay:** 34.1% win rate
- **Switch:** 65.9% win rate

The gap between the two strategies is dramatic and consistent — nearly perfect agreement with the theoretical 1/3 vs. 2/3 split.

---

## Running the Simulation

### Prerequisites

- Python 3.6+
- `matplotlib` — for visualization
- `random` — standard library

### Execution

```bash
python3 code.py
```

The script runs the simulation and generates two PNG files:
- `monty_hall_convergence.png` — convergence plot
- `monty_hall_comparison.png` — final comparison bar chart

---

## Key Insights

1. **Monty's constraint is crucial.** He cannot open a door randomly — he must avoid both your choice and the car. This asymmetry creates the probability shift.

2. **Intuition fails.** Humans tend to interpret Monty's reveal as creating "new symmetry" (two equal doors, each 1/2). In fact, Monty's reveal is *not symmetric* from the game's perspective; it's entirely determined by the true locations of cars and goats.

3. **Conditional probability matters.** The winning strategy depends on what you want to condition on:
   - Conditional on "you initially chose randomly from three doors," switching wins 2/3
   - This holds regardless of which specific door Monty actually opens

4. **Empirical verification is powerful.** While the math is sound, watching 1000 independent trials converge to the theoretical prediction transforms abstract probability into concrete observation.

---

## Project Structure

```
Personal Project 3: Monty Hall Simulation (Goat and Car Game Show)/
|-- code.py                           # Main simulation & visualization
|-- monty_hall_convergence.png        # Convergence plot (output)
|-- monty_hall_comparison.png         # Comparison bar chart (output)
`-- README.md                         # This file
```

---

**References:**
- vos Savant, Marilyn. "Ask Marilyn." *Parade Magazine*, September 1990.
- Selvin, Steve. "A Problem in Probability." *American Statistician*, 1975.

<p align="center">
  <a href="https://nels.buhrley.com/CPP-Numerical-Modeling/#personal-projects"><strong>← Back to Portfolio Hub</strong></a>
</p>