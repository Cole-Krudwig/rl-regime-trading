# Regime-Aware Meta-Control For Deep Reinforcement Learning In Financial Trading

This repository implements a **regime-aware reinforcement learning** framework for single-asset trading.  
An RL _meta-controller_ selects among a small set of interpretable base strategies (momentum, mean reversion, defensive), which then determine allocations between a risky asset and a risk-free asset.

The core pieces are:

- A custom **Gymnasium-compatible trading environment** (`TradingEnv`) with:
  - Explicit wealth dynamics.
  - Constant relative risk aversion (CRRA) utility–based reward.
  - A risk-free asset with configurable rate.
- A **feature engine** that computes diffusion-inspired features (drift, realized volatility, volatility trend, mean-reversion residuals, price velocity).
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

We track portfolio wealth $\( W_t \)$ with:

$$
W\_{t+1} = W_t \left( 1 + \pi_t r_t + (1 - \pi_t) r_f \right),
$$

where:

- $\( \pi_t \)$ is the fraction in the risky asset,
- $\( r_t \)$ is the risky asset’s return,
- $\( r_f \)$ is the per-step risk-free rate.

Preferences are modeled via **CRRA utility**:

$$
U(W) = \frac{W^{1-\gamma}}{1-\gamma}, \quad \gamma > 1,
$$

with $\(\gamma = 2\)$ in the current implementation. The reward is primarily the **change in utility** $\Delta U_t = U(W_{t+1}) - U(W_t)$, optionally combined with a quadratic penalty on large returns to discourage extreme risk-taking.

This makes the environment **HJB-inspired**: we borrow the structure and objective from continuous-time portfolio problems, but learn the policy in **discrete time via deep RL**.

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

## 3. Usage

### 3.1. Training

```bash
python cli.py train --symbol <symbol> --train-split <train/test split (e.g. 0.8)>
```

### 3.2. Evaluation

```bash
python cli.py evaluate --symbol <symbol> --train-split <train/test split (e.g. 0.8)>
```
An evaluation output will look like:

### 3.3 Summary: Test Performance on Final 20%

| Model | Steps |  FinalW   | TotRet% | Sharpe |
|-------|:-----:|----------:|--------:|-------:|
| DQN   |  1275 | 19646.65  |   96.47 |  0.941 |
| PPO   |  1275 | 13518.35  |   35.18 |  0.597 |
| A2C   |  1275 | 16746.36  |   67.46 |  0.681 |


And also include benchmarks based on the naive strategies themselves.

### Demo Video

<video src="https://github.com/user-attachments/assets/3c428a44-0058-448e-adde-f06bc3f6bbe1"><\video>



---

## 4. Repo Structure

```
.
├─ app/
│  ├─ data/
│  │  └─ spy.csv             # Or any compatible data
│  ├─ logs/
│  │  └─ # Generated log files
│  ├─ literature/
│  │  └─ main.pdf           # Arxiv Preprint
│  ├─ models/
│  │  └─ # Model save checkpoints
│  ├─ src/
│  │  ├─ __init__.py
│  │  ├─ BenchmarkEvaluator.py
│  │  ├─ Evaluate.py
│  │  ├─ FeatureEngine.py
│  │  ├─ Fetch.py
│  │  ├─ TradingEnv.py
│  │  ├─ TradingStrategies.py
│  │  └─ Train.py
├─ .gitignore
├─ cli.py
├─ README.md
├─ requirements.txt          # python deps
└─ README.md
```

---

## 5. Citing This Work

If you use this code or framework in academic work, please consider citing the accompanying paper (see main.tex or the literature/ directory). A BibTeX entry can be added once the paper has a finalized venue and citation.

---

## 6. Contact

For questions, issues, or suggestions:
- Open an issue on the repository, or
- Contact directly via email (ckrudwig@gmail.com) or github.
