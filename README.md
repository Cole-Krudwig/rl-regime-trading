# rl-regime-trading

Regime-Aware Reinforcement Learning for Trading. This project combines stochastic calculus–based regime detection with RL meta-policies to adapt trading strategies across bull, bear, and high-volatility market regimes. Includes data pipeline, regime identification, baseline strategies, and RL experiments with performance evaluation.

> **TL;DR**: We fetch market data → build features (momentum / volatility / mean-reversion) → feed a 5×7 observation (shape `(35,)`) → DQN chooses one of 3 strategies each step → reward reflects wealth change via a utility-inspired objective with optional costs/slippage.

---

## ✨ Highlights

- **Three-action policy**: `0=Momentum`, `1=Mean Reversion`, `2=Defensive`
- **Observation**: rolling **lookback of 5** with **7 features** → flattened to **(35, )**
- **Features**: short/long momentum, volatility, mean-reversion signals, plus log-returns (target switch from raw returns)
- **Environment**: Gym-style API (`reset/step`) for plug-and-play with RL libraries
- **Agent**: DQN baseline with target network, ε-greedy, replay buffer
- **Reward**: wealth-change utility (PDE-inspired), supports transaction costs and slippage
- **Extensible**: drop-in new features, alternative agents (PPO/A2C/SAC), and markets

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

## ⚙️ Installation

**Python**: 3.10+ recommended

```bash
# (Option A) Conda
conda create -n phase4 python=3.10 -y
conda activate phase4

# (Option B) venv
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install deps
pip install -r requirements.txt
```

### Environment variables

Create a `.env` (or export in your shell) with your market data provider key(s):

```bash
# .env
ALPHAVANTAGE_API_KEY=YOUR_KEY_HERE
```

The code expects this key when calling the data fetcher.

---

## 📈 Data Pipeline

1. **Collect** market data via `app/tools/data_fetcher.py` (AlphaVantage by default).
2. **Preprocess** and **feature engineer** with `app/tools/features.py`, including:
   - Short/long momentum
   - Volatility proxy
   - Mean-reversion signal(s)
   - Log returns (target variable switch from raw returns)
3. **Assemble observation** with a lookback window of **5** steps × **7** features → `shape=(35,)`.

Example (pseudocode):

```python
from app.tools.data_fetcher import fetch_bars
from app.tools.features import build_features

df = fetch_bars(symbol="SPY", start="2017-01-01", end="2024-12-31", interval="1d")
X, y = build_features(df, lookback=5)  # returns obs matrix and (optional) targets
```

---

## 🧪 Environment

- **Action space**: `Discrete(3)` → `{0: Momentum, 1: Mean Reversion, 2: Defensive}`
- **Observation**: `Box(shape=(35,), dtype=float32)`
- **Episode**: one contiguous time range (train/test split configurable)
- **Reward**: change in portfolio wealth under selected sub-strategy; can include
  - transaction costs
  - slippage
  - risk penalties

```python
import gym
from app.envs.trading_env import TradingEnv

env = TradingEnv(df, lookback=5, features=7, costs=0.0001, slippage=0.0002)
obs = env.reset()
obs, reward, done, info = env.step(env.action_space.sample())
```

---

## 🤖 DQN Agent

- **Network**: MLP with target network
- **Exploration**: ε-greedy with linear decay
- **Replay**: uniform buffer (prioritized optional)
- **Stability knobs**: learning rate, target update τ/period, batch size, γ

```python
from app.agents.dqn import DQNAgent
from app.envs.trading_env import TradingEnv

env = TradingEnv(df, lookback=5, features=7)
agent = DQNAgent(env.observation_space, env.action_space, cfg="configs/dqn.yaml")
agent.train(env, total_steps=200_000)
```

---

## 🚀 Quickstart

### 1) Fetch data (optional if you already have a CSV)

```bash
python -m app.tools.data_fetcher --symbol SPY --start 2017-01-01 --end 2024-12-31 --interval 1d --out data/SPY_1d.csv
```

### 2) Train DQN

```bash
python train.py --config configs/dqn.yaml \
  --symbol SPY \
  --start 2017-01-01 --end 2024-12-31 \
  --interval 1d \
  --seed 42
```

Useful flags (check `train.py -h` for all):

- `--save-dir` (checkpoints)
- `--tensorboard` (enable TB logging)
- `--eval-split` (out-of-sample fraction or date)

### 3) Evaluate / Backtest

```bash
python evaluate.py --checkpoint runs/dqn/SPY/best.pt \
  --symbol SPY --start 2020-01-01 --end 2024-12-31 --interval 1d
```

Outputs: PnL curve, action traces, hit ratios, drawdowns, and basic trade stats.

---

## 🔧 Configuration (example: `configs/dqn.yaml`)

```yaml
seed: 42
env:
  lookback: 5
  features: 7
  transaction_costs: 0.0001
  slippage: 0.0002
agent:
  hidden_sizes: [128, 128]
  gamma: 0.99
  lr: 1.0e-3
  batch_size: 128
  buffer_size: 100000
  start_epsilon: 1.0
  end_epsilon: 0.05
  epsilon_decay_steps: 100000
  target_update_period: 1000
train:
  total_steps: 200000
  eval_every: 5000
  checkpoint_every: 10000
```

---

## 📊 Logging

Enable TensorBoard:

```bash
python train.py --config configs/dqn.yaml --tensorboard
tensorboard --logdir runs/
```

Expect:

- training loss, Q-targets
- ε-schedule, reward distribution
- evaluation metrics

---

## 🧩 Extending the Project

- **Add a strategy**: implement `act(weights, state)` in `app/strategies/your_strategy.py`
- **Swap agent**: add PPO/A2C/SAC under `app/agents/` and a matching `--agent` flag
- **New features**: modify `app/tools/features.py`, update `features` count in config

---

## 🧯 Troubleshooting

- **No API key found**: set `ALPHAVANTAGE_API_KEY` in `.env` or your shell.
- **Shape mismatches**: ensure `lookback × features = 35` (or update config + network).
- **Learning but no edge**: try
  - reward shaping (risk-adjusted returns),
  - costs/slippage tuning,
  - different seeds / train windows,
  - prioritized replay,
  - smaller learning rate / target update frequency.

---

## 🗺️ Roadmap

- ✓ DQN baseline and env
- [ ] Add PPO/A2C baselines
- [ ] Transaction-cost aware training
- [ ] Robust slippage modeling
- [ ] Walk-forward evaluation and cross-validation
- [ ] Model-based RL for regime transitions

---

## 📚 Citation / Acknowledgements

If you use this code in research, please cite this repo. Portions of the environment design and reward shaping are informed by stochastic-process utility formulations discussed in the accompanying slide deck (“Phase 4”).

---

## 📝 License

MIT (see `LICENSE`).
