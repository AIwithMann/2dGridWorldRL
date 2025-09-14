````markdown
# 2D GridWorld Reinforcement Learning

This repository implements a simple **2D GridWorld** environment and agents to practice fundamental **Reinforcement Learning (RL)** algorithms.  
The project is designed for hands-on exploration of RL concepts with a focus on **Monte Carlo** and **SARSA (n-step)** methods.

---

## Files

- **TwoDEnv.py**  
  Defines the `Env2d` environment:
  - A grid with customizable rows and columns.
  - Randomly assigned termination states with reward = 1.
  - Non-terminal states have reward = 0.

- **MonteCarlo.py**  
  Implements a **Monte Carlo control agent** (`MCagent`):
  - Learns a value function `Vgrid`.
  - Improves a deterministic target policy `TPgrid`.
  - Uses exploring starts and weighted importance sampling for updates.
  - Demonstrates on-policy policy iteration.

- **Sarsa.py**  
  Implements a **SARSA n-step agent** (`SarsaAgent`):
  - Learns an action-value function `Qgrid`.
  - Improves policy `Pgrid` based on greedy action selection.
  - Supports n-step backups (parameter `numBackups`).
  - Demonstrates temporal-difference learning.

---

## Algorithms

1. **Monte Carlo Control**
   - Episodic interaction until a terminal state.
   - Updates based on complete returns.
   - Slowly converges to optimal policy with enough episodes.

2. **SARSA (n-step)**
   - Temporal-difference method.
   - Updates based on partial returns with lookahead.
   - More efficient than pure Monte Carlo.

---

## Usage

Run Monte Carlo training:
```bash
python MonteCarlo.py
````

Run SARSA training:

```bash
python Sarsa.py
```

Both scripts will print the evolving **value grid** and **policy grid** during training.

---

## Example Output (Monte Carlo)

```
Terminal states: [(1, 3), (4, 2)]
Value Grid:
[[0.1  0.05 ...]
 ... ]
Policy Grid:
[['↑' '→' ...]
 ... ]
```

---

## Requirements

* Python 3.8+
* NumPy

Install dependencies:

```bash
pip install numpy
```

---

## Notes

* Start with **Monte Carlo** to understand episodic returns.
* Compare with **SARSA** to see the effect of temporal-difference updates.
* You can modify grid size, discount factor (`gamma`), exploration (`epsilon`), and number of iterations.

---

## Future Work

* Add Q-learning.
* Add visualization of the agent’s trajectory.
* Extend environment with negative rewards or obstacles.

```

Do you want me to also include a diagram of the **GridWorld** (as an image in the README) so it’s easier for others to visualize?
```
