# rl-regime-trading

# HJB-Guided Regime RL for Single-Asset Trading

This repository implements a **regime-aware reinforcement learning** framework for single-asset trading.  
An RL _meta-controller_ selects among a small set of interpretable base strategies (momentum, mean reversion, defensive), which then determine allocations between a risky asset and a risk-free asset.

The core pieces are:

- A custom **Gymnasium-compatible trading environment** (`HJBTradingEnv`) with:
  - Explicit wealth dynamics.
  - CRRA (power) utility–based reward.
  - A risk-free asset with configurable rate.
- A **stochastic feature engine** that computes diffusion-inspired features (drift, realized volatility, volatility trend, mean-reversion residuals, price velocity).
- RL agents (DQN, PPO, A2C) trained via **Stable-Baselines3**.
- A command-line interface for:
  - Fetching data.
  - Training models.
  - Evaluating models.
  - Running benchmark strategies and exporting plot-ready CSVs.

This code underpins an accompanying LaTeX research paper on HJB-guided regime meta-control.

---

## 1. Core Ideas

### Regime Meta-Controller

Instead of learning raw portfolio weights directly, the agent chooses among **three base strategies**:

- **Momentum** – full long allocation to the risky asset.
- **Mean Reversion** – allocation determined by standardized residuals of price vs. a rolling mean (betting against extreme deviations).
- **Defensive** – small short allocation and large risk-free allocation for bear / high-volatility regimes.

At each time step, the RL agent outputs an **action** in `{0, 1, 2}`, which maps to one of these strategies. The strategy then produces a portfolio weight in the risky asset and the complementary weight in the risk-free asset.

### Wealth Dynamics & Reward

We track portfolio wealth \( W_t \) with:

$$
W\_{t+1} = W_t \left( 1 + \pi_t r_t + (1 - \pi_t) r_f \right),
$$

where:

- \( \pi_t \) is the fraction in the risky asset,
- \( r_t \) is the risky asset’s return,
- \( r_f \) is the per-step risk-free rate.

Preferences are modeled via **CRRA utility**:

$$
U(W) = \frac{W^{1-\gamma}}{1-\gamma}, \quad \gamma > 1,
$$

with \(\gamma = 2\) in the current implementation. The reward is primarily the **change in utility** \( \Delta U*t = U(W*{t+1}) - U(W_t) \), optionally combined with a quadratic penalty on large returns to discourage extreme risk-taking.

This makes the environment **HJB-inspired**: we borrow the structure and objective from continuous-time HJB portfolio problems, but learn the policy in **discrete time via deep RL**.

---

## 2. Installation

### 2.1. Clone the repository

```bash
git clone <https://github.com/Cole-Krudwig/rl-regime-trading.git>.git
cd rl-regime-trading
```

### 2.2. Create and activate a Python environment

```bash
conda create -n rl-regime-trading python=3.11
conda activate rl-regime-trading
```

or with venv:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 2.3. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies include:

- gymnasium
- stable-baselines3
- pandas
- numpy
- yfinance

---

## 📦 Repo Structure

```
.
├─ app/
│  ├─ data/
│  │  └─ spy.csv             # Or any compatible data
│  ├─ logs/
│  │  └─ # Generated log files
│  ├─ literature/
│  │  └─ Paper.pdf           # Arxiv Preprint
│  ├─ models/
│  │  └─ # Model save checkpoints
│  ├─ src/
│  │  ├─ __init__.py
│  │  ├─ Fetch.py
│  │  ├─ gather_data.ipynb
│  │  ├─ FeatureEngine.py
│  │  ├─ TradingEnv.py
│  │  └─ TradingStrategies.py
├─ evaluate_model.py
├─ train_agent.py
├─ requirements.txt          # python deps
└─ README.md
```

---
