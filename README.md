# Reinforcement Learning on 2D Grid Environment

This repository contains a simple implementation of Reinforcement Learning algorithms on a 2D grid environment. The environment is custom-made, and two learning methods are included:

- **Monte Carlo Control (MC)**
- **n-step Sarsa**

The project is built for hands-on practice with RL concepts.

---

## Files

- **`TwoDEnv.py`**  
  Defines the 2D grid environment (`Env2d`).  
  - Each cell is either a normal state or a terminal state.  
  - Rewards are placed randomly in terminal states.  

- **`MonteCarlo.py`**  
  Implements an on-policy Monte Carlo control agent.  
  - Learns a state-value function (`Vgrid`) and updates a deterministic greedy policy (`TPgrid`).  
  - Uses weighted importance sampling for updates.  

- **`Sarsa.py`**  
  Implements an n-step Sarsa agent.  
  - Learns an action-value function (`Qgrid`) and improves the policy (`Pgrid`) iteratively.  
  - Balances exploration and exploitation with epsilon-greedy behavior.  
  - Supports configurable step size `n`.  

---

## Algorithms

### Monte Carlo Agent
- Uses **episode-based sampling**.  
- Starts from a random state, follows epsilon-greedy behavior until termination.  
- Updates state-value function using weighted importance sampling.  
- Improves policy by making it greedy w.r.t. updated values.  

### n-step Sarsa Agent
- Uses **temporal difference learning with backups of length `n`**.  
- At each step:  
  - Updates `Q(s,a)` using observed rewards and bootstrapped estimates.  
  - Improves the policy grid (`Pgrid`) to be greedy w.r.t. learned action-values.  
- Parameter `n` controls bias-variance tradeoff.  

---

## Example Usage

### Monte Carlo
```bash
python MonteCarlo.py
````

Sample output (value grid and policy grid evolve over iterations):

```
Value Grid:
[[...]]
Policy Grid:
[['→' '↑' ...]
 [...]]
```

### Sarsa

```bash
python Sarsa.py
```

Sample output (policy grid shown every 100 episodes):

```
Episode 100/1000
Policy grid:
[['↑' '→' ...]
 [...]]
```

---

## Parameters

* `gamma` (float): Discount factor.
* `epsilon` (float): Probability of random action (exploration).
* `maxIterations` (int): Number of training episodes.
* `numBackups` (int, Sarsa only): Number of steps (`n`) in n-step updates.
* `alpha` (float, Sarsa only): Learning rate.

---

## Key Differences

* **MC**: Updates only after an entire episode. Works well with complete trajectories.
* **Sarsa**: Updates online with n-step lookahead. More sample-efficient.

---

## Notes

* The code is educational and not optimized.
* For reproducibility, random seeds can be fixed before running.
* Currently, rewards are binary: `1` at terminal states, `0` elsewhere.

---
