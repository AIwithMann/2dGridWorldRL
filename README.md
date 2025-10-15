# Reinforcement Learning on 2D Grid Environment

This repository contains a simple implementation of Reinforcement Learning algorithms on a 2D grid environment. The environment is custom-made, and several learning methods are included:

- **Monte Carlo Control (MC)**
- **n-step Sarsa**
- **1-step Q-learning**
- **n-step Q-learning**
- **Sarsa(λ)**
- **Watkins's Q(λ)**
- **Dyna-Q** (Q-learning with planning steps)

The project is built for hands-on practice with RL concepts.

---

## Files

- **`TwoDEnv.py`**  
  Defines the 2D grid environment (`Env2d`).  
  - Each cell is either a normal state or a terminal state.  
  - Rewards are placed at terminal states.  

- **`MonteCarlo.py`**  
  Implements an on-policy Monte Carlo control agent.  
  - Learns a state-value function (`Vgrid`) and updates a deterministic greedy policy (`TPgrid`).  
  - Uses weighted importance sampling for updates.  

- **`Sarsa.py`**  
  Implements an n-step Sarsa agent.  
  - Learns an action-value function (`Qgrid`) and improves the policy (`Pgrid`) iteratively.  
  - Balances exploration and exploitation with epsilon-greedy behavior.  
  - Supports configurable step size `n`.  

- **`OneStepQLearning.py`**  
  Implements 1-step Q-learning agent.  
  - Learns an action-value function (`Qgrid`) and updates the target policy (`TPgrid`).  
  - Uses epsilon-greedy behavior and a diminishing epsilon schedule.  

- **`nStepQLearning.py`**  
  Implements n-step Q-learning agent.  
  - Learns an action-value function (`Qgrid`) and updates the target policy (`TPgrid`).  
  - Balances exploration and exploitation with epsilon-greedy behavior.  
  - Supports n-step updates with optional diminishing alpha (learning rate).  

- **`Sarsa(λ).py`**  
  Implements Sarsa(λ) agent (on-policy TD(λ)).  
  - Learns an action-value function and updates the policy.  
  - Uses eligibility traces to speed up learning.  
  - Balances exploration and exploitation with epsilon-greedy behavior and decaying epsilon.  

- **`WatkinsQ(λ).py`**  
  Implements Watkins's Q(λ) agent (off-policy TD(λ)).  
  - Learns an action-value function and updates the policy.  
  - Uses eligibility traces for faster convergence.  
  - Exploration handled with epsilon-greedy implicit policy.  

- **`DynaQ.py`**  
  Implements the **Dyna-Q algorithm**: Q-learning with planning steps.  
  - Learns an action-value function (`Qgrid`) and updates the policy (`TPgrid`).  
  - Maintains a model of the environment (`model`) to simulate experiences.  
  - Performs `n` planning steps per real step to accelerate learning.  
  - Balances exploration and exploitation with epsilon-greedy behavior and decaying epsilon.  

---

## Algorithms

### Monte Carlo Agent
- Uses **episode-based sampling**.  
- Starts from a random state, follows epsilon-greedy behavior until termination.  
- Updates state-value function using weighted importance sampling.  
- Improves policy by making it greedy w.r.t. updated values.  

### n-step Sarsa Agent
- Uses **on-policy TD learning with n-step backups**.  
- Updates `Q(s,a)` using observed rewards and bootstrapped estimates.  
- Improves the policy grid (`Pgrid`) to be greedy w.r.t. learned action-values.  
- Parameter `n` controls bias-variance tradeoff.  

### 1-step Q-learning Agent
- Uses **off-policy TD learning with 1-step backups**.  
- Updates `Q(s,a)` using observed rewards and `max_a Q(s',a)`.  
- Target policy (`TPgrid`) is greedy, behavior policy (`Pgrid`) is random.  

### n-step Q-learning Agent
- Uses **off-policy TD learning with n-step backups**.  
- Updates `Q(s,a)` using observed rewards and `max_a Q(S_{t+n},a)`.  
- Target policy (`TPgrid`) is greedy, behavior policy is epsilon-greedy.  

### Sarsa(λ) Agent
- Uses **on-policy TD(λ)** with eligibility traces.  
- Updates `Q(s,a)` using traces and next state-action values.  
- Policy (`Pgrid`) is greedy w.r.t. learned Q-values.  

### Watkins's Q(λ) Agent
- Uses **off-policy TD(λ)** with eligibility traces.  
- Updates `Q(s,a)` using traces and next state-action values.  
- Policy is greedy (`TPgrid`), behavior is epsilon-greedy.  

### Dyna-Q Agent
- Combines **Q-learning** with **planning**.  
- Maintains a model of transitions and rewards (`model`).  
- Performs multiple simulated updates per real step.  
- Accelerates learning by reusing past experiences.  
- Policy (`TPgrid`) is greedy, behavior is epsilon-greedy.  

---

## Parameters

* `gamma` (float): Discount factor.  
* `epsilon` (float): Probability of random action (exploration).  
* `maxIterations` (int): Number of training episodes.  
* `numBackups` (int, Sarsa only): Number of steps (`n`) in n-step updates.  
* `alpha` (float, Sarsa/Q-learning only): Learning rate.  
* `planningSteps` (int, Dyna-Q only): Number of simulated updates per real step.  

---

## Key Differences

* **MC**: Updates only after an entire episode; works well with complete trajectories.  
* **Sarsa**: Updates online with n-step lookahead; more sample-efficient.  
* **1-step Q-learning**: Updates online with 1-step lookahead; behavior policy is not updated.  
* **n-step Q-learning**: Updates online with n-step lookahead; behavior policy is epsilon-greedy but not changed.  
* **Sarsa(λ)**: Updates online with eligibility traces; faster convergence but higher space/time complexity.  
* **Watkins's Q(λ)**: Off-policy TD(λ); faster convergence with eligibility traces, higher space/time complexity.  
* **Dyna-Q**: Combines Q-learning and planning; accelerates learning by simulating multiple experiences per real step.  

---

## Notes

* The code is educational and not optimized for speed.  
* Random seeds can be fixed for reproducibility.  
* Currently, rewards are binary: `1` at terminal states, `0` elsewhere.  
