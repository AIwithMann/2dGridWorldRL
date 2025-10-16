# Reinforcement Learning on 2D Grid Environment

This repository contains a set of implementations of **Reinforcement Learning (RL) algorithms** applied to a **2D grid environment**. The environment is custom-made, and the project is intended for hands-on practice with RL concepts and algorithms.

---

## Environment

- **`TwoDEnv.py`**  
  Defines the 2D grid environment (`Env2d`).  
  - Each cell is either a normal state or a terminal state.  
  - Rewards are assigned to terminal states.  
  - Supports configurable grid size and multiple terminal states.  

---

## Implemented Algorithms

The repository includes several RL agents:

1. **Monte Carlo Control (MC)**  
2. **n-step Sarsa**  
3. **1-step Q-learning**  
4. **n-step Q-learning**  
5. **Sarsa(λ)**  
6. **Watkins's Q(λ)**  
7. **Dyna-Q** (Q-learning with planning)  
8. **Dyna-Q with Prioritized Sweeping**  

Each agent learns an **action-value function (`Qgrid`)** or **state-value function (`Vgrid`)** and updates a corresponding policy (`Pgrid` or `TPgrid`).  

---

## Files and Descriptions

| File | Description |
|------|-------------|
| `TwoDEnv.py` | Defines the 2D grid world. |
| `MonteCarlo.py` | On-policy MC control; updates `Vgrid` and greedy policy `TPgrid`. Uses weighted importance sampling. |
| `Sarsa.py` | n-step Sarsa agent; updates `Qgrid` and policy `Pgrid` using on-policy TD updates. |
| `OneStepQLearning.py` | 1-step off-policy Q-learning; updates `Qgrid` and target policy `TPgrid`. |
| `nStepQLearning.py` | n-step off-policy Q-learning; updates `Qgrid` and `TPgrid` using n-step backups. |
| `Sarsa(λ).py` | On-policy Sarsa(λ) with eligibility traces; updates `Qgrid` and `Pgrid`. |
| `WatkinsQ(λ).py` | Off-policy Q(λ) agent; uses eligibility traces to accelerate learning. |
| `DynaQ.py` | Dyna-Q agent: combines Q-learning with planning using a model of transitions and rewards. |
| `DynaQ-PS.py` | Dyna-Q with **Prioritized Sweeping**: planning updates prioritized by TD-error. Accelerates learning by focusing on important transitions. |

---

## Algorithm Highlights

### Monte Carlo Control
- Episode-based updates; policy improved after each episode.  
- Weighted importance sampling for accurate value estimation.  
- Behavior policy is ε-greedy.

### n-step Sarsa
- On-policy TD learning with `n`-step lookahead.  
- Updates `Q(s,a)` and policy `Pgrid`.  
- Parameter `n` controls bias-variance trade-off.

### 1-step Q-learning
- Off-policy TD update with 1-step backup.  
- Target policy `TPgrid` is greedy; behavior policy is ε-greedy.

### n-step Q-learning
- Off-policy TD updates using `n` steps.  
- Balances exploration and exploitation with ε-greedy behavior.

### Sarsa(λ)
- On-policy TD(λ) with eligibility traces.  
- Faster convergence than n-step methods.  
- Policy updated greedily with respect to Q-values.

### Watkins’s Q(λ)
- Off-policy TD(λ) with eligibility traces.  
- Target policy is greedy, behavior is ε-greedy.

### Dyna-Q
- Q-learning combined with **planning steps** using a learned model.  
- Each real step is followed by `n` simulated updates to accelerate learning.  
- Policy is greedy w.r.t. `Qgrid`, behavior is ε-greedy.

### Dyna-Q with Prioritized Sweeping
- Prioritizes **planning updates** using TD-error.  
- Updates the most significant state-action pairs first.  
- Significantly improves efficiency in environments with sparse rewards.  

---

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `gamma` | Discount factor (0 ≤ γ ≤ 1) |
| `epsilon` | Probability of random exploration |
| `alpha` | Learning rate for Q/Sarsa updates |
| `maxIterations` | Number of training episodes |
| `n` | Number of steps for n-step Sarsa/Q-learning |
| `planningSteps` | Number of simulated updates per real step (Dyna-Q) |
| `theta` | Minimum TD-error threshold for prioritized sweeping |

---

## Notes

- The code is primarily educational and **not optimized for speed**.  
- Rewards are usually binary: `1` at terminal states, `0` elsewhere.  
- Random seeds can be fixed for reproducibility.  
- Prioritized Sweeping may slow down if θ is too small due to large priority queue size; use a heap-based queue for efficiency.  

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd edition). MIT Press.  
