
# 2D Gridworld Reinforcement Learning Agent

This repository contains a Python implementation of a **grid-based environment** and a **reinforcement learning agent** that learns an optimal policy using **importance sampling with Monte Carlo policy evaluation**. The agent interacts with the environment, collects episodes, and updates its value function and policy over time.

---

## Table of Contents

* [Project Overview](#project-overview)
* [Environment](#environment)
* [Agent](#agent)
* [Features](#features)
* [Usage](#usage)
* [Example](#example)
* [Dependencies](#dependencies)
* [License](#license)

---

## Project Overview

The project demonstrates:

* Creation of a gridworld environment with terminal states.
* Monte Carlo reinforcement learning to estimate the state-value function (`Vgrid`).
* Policy improvement via a greedy target policy (`TPgrid`).
* Episode generation and backward updates for importance sampling.

The agent learns the optimal policy for reaching terminal states with maximum reward.

---

## Environment

The environment (`Env`) is defined as a 2D grid of size `rows x cols`:

* **Reward Grid (`Rgrid`)**: Initialized with zeros and randomly assigned `1` for terminal states.
* **Termination States**: Cells with reward `1` are terminal.
* Randomly selects positions for terminal states at initialization.

**Example:**

```python
env = Env(rows=5, cols=5, termitationStates=2)
```

---

## Agent

The agent (`Agent`) interacts with the environment using the following features:

* **State-value grid (`Vgrid`)**: Stores estimated values for each state.
* **Target policy grid (`TPgrid`)**: Stores the greedy action for each state.
* **Behavior policy (`BP`)**: Initially uniform random policy.
* **Epsilon-greedy exploration**: Chooses greedy action with probability `1-epsilon` and random action otherwise.
* **Episode-based learning**: Updates value function and policy after each episode.
* **Monte Carlo importance sampling**: Performs backward updates to correct for off-policy learning.

**Actions:** Left (`←`), Up (`↑`), Right (`→`), Down (`↓`)

---

## Features

* **Random terminal states**
* **Epsilon-greedy action selection**
* **Monte Carlo policy evaluation with importance sampling**
* **Greedy policy improvement**
* **Boundary-safe movements**

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/AIwithMann/2dGridWorldRL.git
cd 2dGridWorldRL
```

2. Install dependencies:

```bash
pip install numpy
```

3. Run the agent:

```python
python main.py
```

---

## Example

```python
env = Env(5, 5, 2)
agent = Agent(env, gamma=0.8, epsilon=0.1, maxIterations=20)
agent.train()
```

**Output:**

```
Terminal states: [(np.int64(1), np.int64(0)), (np.int64(3), np.int64(2))]
Value Grid:
[[ 0.9936097   0.99029185  0.98300543  0.98004571  0.99547055]
 [ 0.96708664  0.99969992  0.96584621  1.00464839  0.99199519]
 [ 0.99987807  0.9999344   0.99983647  0.9999689   0.99996864]
 [ 0.99580006  0.99580248  0.99579026  0.99577724  0.99578193]
 [ 1.05039344  1.16373353  0.42424311 -0.11235044 -1.75173592]]
Policy Grid:
[['↓' '↓' '↓' '↓' '↓']
 ['↓' '↓' '↓' '↓' '↓']
 ['↑' '↑' '↑' '↑' '↑']
 ['↑' '↑' '↑' '↑' '↑']
 ['↑' '↑' '←' '←' '←']]
```

---

## Dependencies

* Python 3.8+
* NumPy

---

## License

This project is licensed under the MIT License.


