# Reinforcement Learning on 2D Grid Environment

This repository contains a simple implementation of Reinforcement Learning algorithms on a 2D grid environment. The environment is custom-made, and two learning methods are included:

- **Monte Carlo Control (MC)**
- **n-step Sarsa**
- **1-step Q-learning**
- **n-step Q-learning**
- **Sarsa(λ)**

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
  - 
- **`OneStepQLearning.py`**
  Impements 1-step Q learning agent.
  - Learns an action value function (`Qgrid`) and improvees the policy (`TPgrid`) while following the random policy (`Pgrid`)
  - Balances exploratioon and exploitation with epsilon-greedy behavior and diminishing epsilon value.
  - 
- **`nStepQLearning.py`**
  Implements n-step Q learning agent.
  - Learns an action value function (`Qgrid`) and improves the policy (`TPgrid`) while following the random policy.
  - Balanes exploration and exploitation with epsilon-greedy behavior and diminishing epsilon value.
  - Balances the updates with diminishing alpha value.

- **`Sarsa(λ).py`**
  Implements Sarsa(λ) algorithm agent.
  - Learns an action value function function and improves the policy.
  - Balances exploration and exploitation with epsilon-greed behavior and diminishing epsilon value.
  
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

### 1-step Q-learinng Agent
- Uses **off-policy temporal difference learning with 1-step backups**
- At each step:
  - Updates `Q(s,a)` with observed rewards and $$\max_a Q(s',a)$$.
  - Improves the target policy grid (`TPgrid`) to be greedy w.r.t. learned action value function but behavior is handeled by random behavior policy (`Pgrid`) which is not updated

### n-Step Q Learning Agent
- Uses **off-policy temporal difference learning with n-step backups**
- At each step:
  - Updates `Q(s,a)` with observed rewards and $$\max_a Q(S_{t+n},a)$$
  - Improves the target policy grid (`TPgrid`) to be greedy w.r.t. learned action value functions but behavior is handeled by epsilon-greedy implicit policy.

### Sarsa(λ) Agents
- Uses **on-policy TD(λ)** which is also known as **Sarsa(λ)**
- Updates `Q(s,a)` with observed rewards, the eligibility trace for that state-action pair and value function of next state and action pair.
- Imporves the policy grid to be gredy w.r.t. learned action value functions.

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
* **1-step Q-learning**: Updates online with 1-step lookahead. But the policy used for generating episodes is never changed.
* **n-step Q-learning**: Updates online with n-step lookahead. But the policy used for generating episodes is never changed
* **Sarsa(λ)**: Updates online with eligiblity traces. Converges fastly but time and space complexities are a bit higher.
---

## Notes

* The code is educational and not optimized.
* For reproducibility, random seeds can be fixed before running.
* Currently, rewards are binary: `1` at terminal states, `0` elsewhere.

---
