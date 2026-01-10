# 🧠 Brain-Behavior Mapping: Complete Documentation

> A Neuro-Inspired Simulation Framework for Cognitive & Behavioral Analyses

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Command Reference](#command-reference)
5. [Streamlit Dashboard Guide](#streamlit-dashboard-guide)
6. [Module Documentation](#module-documentation)
7. [Scientific Background](#scientific-background)
8. [Team Contributions](#team-contributions)

---

## Project Overview

This project simulates key human brain mechanisms to create a closed-loop brain-behavior system:

- **Motor Imagery Decoding**: Translate "imagined movements" into discrete actions (left/right)
- **Dopamine-Driven Learning**: Model reward learning via Reward Prediction Error (RPE)
- **Reversal Learning**: Test cognitive flexibility when reward contingencies change
- **Explainability**: Visualize which brain regions drive decisions

### Use Case

A "mind-controlled gambling task" where:
1. The BCI decoder interprets motor imagery (simulated EEG)
2. The agent chooses left or right arm in a 2-armed bandit task
3. Dopamine signals (RPE) drive learning based on outcomes
4. At the reversal point, reward probabilities flip
5. The agent must adapt its strategy (cognitive flexibility)

---

## Architecture

```
bio-inspired-bci-rl-agent/
│
├── models/                     # Neural network models
│   ├── bci_decoder.py          # BCI motor imagery decoder (Person 1)
│   └── rpe_agent.py            # Actor-Critic RL agent with RPE (Person 2)
│
├── envs/                       # Gymnasium environments
│   └── gambling_task.py        # 2-armed bandit with reversal (Person 2)
│
├── demos/                      # Runnable applications
│   ├── run_simulation.py       # CLI simulation script (Person 3)
│   └── streamlit_app.py        # Interactive dashboard (Person 3)
│
├── utils/                      # Utility modules
│   ├── viz.py                  # Visualization functions (Person 3)
│   └── xai.py                  # Explainability tools (Person 3)
│
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks
├── docs/                       # Documentation
├── requirements.txt            # Dependencies
└── README.md                   # Project readme
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Git

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/Ismailea4/bio-inspired-bci-rl-agent.git
cd bio-inspired-bci-rl-agent

# 2. Create virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.21.0 | Numerical computing |
| matplotlib | >= 3.5.0 | Static plotting |
| plotly | >= 5.0.0 | Interactive charts |
| streamlit | >= 1.20.0 | Dashboard UI |
| gymnasium | >= 0.28.0 | RL environment |
| pandas | >= 1.3.0 | Data handling |
| tensorflow | >= 2.10.0 | Neural networks (optional) |

---

## Command Reference

### 1. Run Simulation (CLI)

The main simulation script with full command-line options:

```bash
python demos/run_simulation.py [OPTIONS]
```

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--trials` | 500 | Number of trials to run |
| `--reversal` | 250 | Trial number when probabilities reverse |
| `--p-left` | 0.4 | Reward probability for left arm (pre-reversal) |
| `--p-right` | 0.6 | Reward probability for right arm (pre-reversal) |
| `--lr` | 0.02 | Agent learning rate |
| `--gamma` | 0.95 | Discount factor for future rewards |
| `--seed` | 42 | Random seed for reproducibility |
| `--plot` | False | Show visualization plots |
| `--save-plots` | None | Directory to save plot images |
| `--save-results` | None | Path to save results pickle file |
| `--verbose` | False | Enable detailed output |

#### Examples

```bash
# Basic run with default settings
python demos/run_simulation.py

# Custom experiment with plots
python demos/run_simulation.py --trials 1000 --reversal 500 --plot

# High learning rate experiment
python demos/run_simulation.py --lr 0.1 --trials 300 --plot

# Save results for later analysis
python demos/run_simulation.py --save-results results/experiment1.pkl

# Full verbose run with all outputs
python demos/run_simulation.py --trials 800 --reversal 400 --plot --save-plots results/ --verbose
```

### 2. Launch Dashboard

Start the interactive Streamlit dashboard:

```bash
streamlit run demos/streamlit_app.py
```

Opens automatically at: `http://localhost:8501`

### 3. Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_environment.py -v
```

---

## Streamlit Dashboard Guide

### Overview

The dashboard provides real-time visualization of the brain-behavior simulation with these components:

### Interface Layout

```
┌─────────────────────────────────────────────────────────────────┐
│                    🧠 Brain-Behavior Mapping Dashboard           │
├─────────────────────────────────────────────────────────────────┤
│ SIDEBAR                          │  MAIN AREA                   │
│ ┌─────────────────────┐          │  ┌─────────────────────────┐ │
│ │ ⚙️ Simulation       │          │  │ 📍 Current State        │ │
│ │    Settings         │          │  │ Progress | Intention |  │ │
│ │                     │          │  │ RPE | Success Rate     │ │
│ │ - Total Trials      │          │  └─────────────────────────┘ │
│ │ - Reversal Point    │          │                              │
│ │ - P(Right) wins     │          │  ┌───────┬───────┬─────────┐ │
│ │                     │          │  │ Brain │Policy │Dopamine │ │
│ │ 🧠 Agent Parameters │          │  │ Topo  │ Pie   │ Gauge   │ │
│ │ - Learning Rate     │          │  └───────┴───────┴─────────┘ │
│ │ - Discount Factor   │          │                              │
│ │ - Random Seed       │          │  ┌─────────────────────────┐ │
│ │                     │          │  │ ⚡ RPE History (Live)   │ │
│ │ 🎮 Controls         │          │  └─────────────────────────┘ │
│ │ [Reset] [Step]      │          │                              │
│ │                     │          │  ┌─────────────────────────┐ │
│ │ 🎬 Watch Mode       │          │  │ 📈 Success Rate Trend   │ │
│ │ Speed: [Slow/Med/   │          │  └─────────────────────────┘ │
│ │        Fast]        │          │                              │
│ │ [Watch] [Stop]      │          │  ┌─────────────────────────┐ │
│ │                     │          │  │ 🔍 What's Happening?    │ │
│ │ [Run 50 Trials]     │          │  │ (Interpretations)       │ │
│ └─────────────────────┘          │  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Sidebar Controls

#### Simulation Settings

| Control | Range | Description |
|---------|-------|-------------|
| Total Trials | 200-1000 | Number of decisions the agent makes |
| Reversal Point | 100 to (trials-50) | When reward probabilities flip |
| P(Right) wins | 0.5-0.8 | Pre-reversal winning probability |

#### Agent Parameters

| Control | Range | Description |
|---------|-------|-------------|
| Learning Rate | 0.005-0.1 | How fast the agent learns |
| Discount Factor | 0.9-0.99 | Weight of future rewards |
| Random Seed | 0-999 | For reproducible experiments |

#### Control Buttons

| Button | Action |
|--------|--------|
| 🔄 Reset | Restart simulation with current settings |
| ▶️ Step | Execute one trial manually |
| ▶️ Watch | Auto-play simulation |
| ⏹️ Stop | Pause watch mode |
| ⏩ Run 50 Trials | Quick batch execution |

#### Watch Mode Speeds

| Speed | Duration | Delay per Trial |
|-------|----------|-----------------|
| Slow | ~2 min | 120ms |
| Medium | ~1 min | 60ms |
| Fast | ~30s | 30ms |

### Main Display Components

#### 1. Current State Metrics

Four key metrics updated in real-time:

- **Progress**: Trial count / Total trials
- **Decoded Intention**: Last BCI decoded action (👈 Left / 👉 Right)
- **Last RPE**: Dopamine signal value (🟢 positive / 🔴 negative)
- **Success Rate**: Cumulative reward percentage

#### 2. Brain Topography (XAI Visualization)

Shows simulated EEG channel activation:

- **Head shape** with ears and nose for orientation
- **22 EEG channels** from 10-20 system
- **Color intensity** = Channel importance (yellow to red)
- **Highlighted regions**: Left/Right motor cortex
- **Legend**: Shows which hemisphere controls which hand

Anatomical markers:
- FRONT/BACK labels
- L/R hemisphere indicators
- Motor cortex regions shaded when active

#### 3. Policy Pie Chart

Shows the agent's current decision strategy:
- **← Left**: Probability of choosing left
- **Right →**: Probability of choosing right
- Updates after each trial as learning progresses

#### 4. Dopamine Gauge

Visual RPE meter:
- **Range**: -2 to +2
- **Green zone**: Positive RPE (better than expected)
- **Red zone**: Negative RPE (worse than expected)
- **Black line**: Zero reference (expectations met)

#### 5. RPE History Chart

Live bar chart showing:
- **Green bars**: Positive RPE (reward > expected)
- **Red bars**: Negative RPE (reward < expected)
- **Purple dashed line**: Reversal point marker
- **Last 100 trials** displayed for clarity

#### 6. Success Rate Trend

Rolling average of rewards:
- **Orange line**: 20-trial moving average
- **Green dashed**: Optimal performance (60%)
- **Gray dotted**: Chance level (50%)

#### 7. Interpretation Panel

Human-readable explanations:
- **Phase indicator**: Pre-Reversal / Adaptation / Post-Reversal
- **RPE meaning**: What the dopamine signal indicates
- **Policy status**: Exploring vs. committed preference
- **Performance assessment**: Above/below chance

#### 8. Session Statistics

Three-column summary:
- **Overall**: Total trials, rewards, mean RPE
- **Pre-Reversal**: Success rate, action distribution
- **Post-Reversal**: Adaptation progress

---

## Module Documentation

### models/bci_decoder.py (Person 1)

BCI Motor Imagery Decoder using ShallowConvNet architecture.

```python
from models.bci_decoder import BCIDecoder

decoder = BCIDecoder(n_channels=22, n_classes=2)
intention = decoder.decode(eeg_signal)  # Returns 0 (left) or 1 (right)
```

Key features:
- Trained on BNCI2014001 dataset (motor imagery)
- 80% classification accuracy
- Supports left/right hand imagery

### models/rpe_agent.py (Person 2)

Actor-Critic RL agent with explicit RPE computation.

```python
from models.rpe_agent import ActorCriticAgent

agent = ActorCriticAgent(
    state_size=10,
    action_size=2,
    learning_rate=0.02,
    gamma=0.95
)

action = agent.get_action(state)
rpe = agent.update(state, action, reward, next_state, done)
```

Key features:
- Separate actor (policy) and critic (value) networks
- Explicit RPE: δ = r + γV(s') - V(s)
- Softmax policy for exploration

### envs/gambling_task.py (Person 2)

Gymnasium-compatible 2-armed bandit with reversal.

```python
from envs.gambling_task import GamblingTaskEnv

env = GamblingTaskEnv(
    p_left=0.4,
    p_right=0.6,
    reversal_trial=250
)

state = env.reset()
next_state, reward, done, info = env.step(action)
```

Key features:
- Probabilistic rewards (0 or 1)
- Automatic reversal at specified trial
- Tracks trial history

### utils/viz.py (Person 3)

Visualization utilities for analysis.

```python
from utils.viz import plot_rpe_dynamics, plot_reversal_learning

plot_rpe_dynamics(rpe_history, reversal_trial)
plot_reversal_learning(policy_history, reversal_trial)
```

Available functions:
- `plot_rpe_dynamics()`: RPE over time
- `plot_reversal_learning()`: Policy adaptation
- `plot_reward_history()`: Cumulative rewards
- `plot_policy_evolution()`: Decision probabilities

### utils/xai.py (Person 3)

Explainability tools for BCI model.

```python
from utils.xai import compute_gradient_importance, plot_channel_importance

importance = compute_gradient_importance(model, input_signal)
plot_channel_importance(importance, channel_names)
```

---

## Scientific Background

### Reward Prediction Error (RPE)

The dopamine signal that drives learning:

```
RPE = Actual Reward - Expected Reward
    = r + γV(s') - V(s)
```

Where:
- `r` = Received reward (0 or 1)
- `γ` = Discount factor (how much future matters)
- `V(s)` = Value of current state
- `V(s')` = Value of next state

### Biological Basis

| Component | Brain Region | Model Equivalent |
|-----------|--------------|------------------|
| Motor imagery | Motor cortex (C3/C4) | BCI Decoder |
| Reward processing | Ventral striatum | Critic network |
| Action selection | Prefrontal cortex | Actor network |
| Learning signal | VTA dopamine | RPE computation |

### Reversal Learning

Tests cognitive flexibility:
1. **Acquisition**: Learn that Right > Left (60% vs 40%)
2. **Reversal**: Probabilities flip (Left > Right)
3. **Re-learning**: Adapt strategy based on negative RPE

---

## Team Contributions

### Person 1: BCI & Neural Decoding
- ShallowConvNet implementation
- MOABB dataset integration
- Motor imagery classification

### Person 2: RL Environment & Agent
- GamblingTaskEnv with reversal
- Actor-Critic architecture
- Explicit RPE computation

### Person 3: Integration, UX & Explainability
- run_simulation.py CLI tool
- Streamlit dashboard
- Visualization utilities
- XAI brain topography
- Documentation

---

## License

MIT License - See LICENSE file for details.

---

## References

1. Schultz, W. (1997). Dopamine neurons and their role in reward mechanisms.
2. Sutton & Barto (2018). Reinforcement Learning: An Introduction.
3. Schirrmeister et al. (2017). Deep learning with convolutional neural networks for EEG decoding.
