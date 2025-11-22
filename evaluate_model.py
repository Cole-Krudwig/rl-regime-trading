# evaluate_model.py

import os
import numpy as np
import pandas as pd

from stable_baselines3 import DQN

# ---- Adjust these imports to match your project structure ----
from src.TradingEnv import HJBTradingEnv
from src.StochasticFeatureEngine import StochasticFeatureEngine
from src.TradingStrategies import TradingStrategies

# Same paths as in train_agent.py (adjust if needed)
DATA_PATH = "data/spy.csv"
# <-- change to your actual model file
MODEL_PATH = "models/dqn_hjb_final.zip"


def build_test_data() -> pd.DataFrame:
    """
    Rebuilds the exact same data pipeline as train_agent.make_trading_env,
    then returns only the final 20% slice used for evaluation.
    """
    df = pd.read_csv(DATA_PATH, index_col=0)

    # Feature engineering
    engine = StochasticFeatureEngine(df)
    processed_df = engine.calculate_all_features()

    merged_df = pd.merge(
        df,
        processed_df,
        left_index=True,
        right_index=True,
        how="inner",
    )

    # Train / test split (same 80/20 as in train_agent.py)
    split_idx = int(0.8 * len(merged_df))
    test_df = merged_df.iloc[split_idx:].copy().reset_index(drop=True)
    return test_df


def evaluate_model(model_path: str) -> None:
    """
    Evaluates a trained DQN model on the final 20% of the data (one pass).

    Tracks:
      - total portfolio return
      - final wealth
      - annualized Sharpe ratio
    """
    # Load trained model
    model = DQN.load(model_path)

    # Build test data slice
    test_df = build_test_data()

    # Pull constants from env class
    LOOKBACK_STEPS = HJBTradingEnv.LOOKBACK_STEPS
    FEATURE_COLS = HJBTradingEnv.FEATURE_COLS
    INITIAL_WEALTH = HJBTradingEnv.INITIAL_WEALTH

    # Strategies object uses the same data slice as the environment would
    strategies = TradingStrategies(test_df)

    # ----- Helper to build base observation (one time step) -----
    def make_base_obs(t_idx: int, wealth: float) -> np.ndarray:
        """
        Mimics HJBTradingEnv._get_base_obs:
        [ stochastic Z-features at t, normalized wealth ]
        """
        features = test_df.loc[t_idx, FEATURE_COLS].values.astype(np.float32)
        norm_wealth = np.float32(wealth / INITIAL_WEALTH)
        base = np.concatenate([features, [norm_wealth]]).astype(np.float32)
        return base

    # ----- Evaluation rollout over the test slice -----

    # Start wealth
    wealth = INITIAL_WEALTH
    portfolio_returns = []

    # We mimic env.reset(): current_step = t0,
    # obs_history = [base_obs(t0)] * LOOKBACK_STEPS
    # Then each action is applied to the NEXT day (t0+1, t0+2, ...)
    t0 = LOOKBACK_STEPS  # burn-in index inside test slice
    if t0 >= len(test_df) - 1:
        raise ValueError(
            "Test set too small for evaluation with given LOOKBACK_STEPS.")

    base_obs = make_base_obs(t0, wealth)
    obs_history = [base_obs for _ in range(LOOKBACK_STEPS)]
    current_t = t0

    # Loop until we run out of test data
    while True:
        # Build stacked observation [x_{t-L+1}, ..., x_t]
        obs = np.concatenate(obs_history, axis=0).astype(np.float32)

        # DQN expects shape (obs_dim,) or (1, obs_dim); this works fine
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # Environment step would increment current_step first
        next_t = current_t + 1
        if next_t >= len(test_df):
            break  # no more data to step into

        # Compute portfolio return for this step using the strategy module
        portfolio_return = strategies.calculate_strategy_returns(
            next_t, action)
        portfolio_returns.append(portfolio_return)

        # Update wealth
        wealth *= (1.0 + portfolio_return)

        # Build next base obs with updated wealth and features at next_t
        next_base_obs = make_base_obs(next_t, wealth)
        obs_history.pop(0)
        obs_history.append(next_base_obs)

        current_t = next_t

    # ----- Metrics -----
    final_wealth = wealth
    total_return = final_wealth / INITIAL_WEALTH - 1.0

    returns_arr = np.array(portfolio_returns, dtype=np.float64)
    mean_ret = returns_arr.mean()
    std_ret = returns_arr.std(ddof=1) if len(returns_arr) > 1 else np.nan

    # Annualized Sharpe assuming ~252 trading days
    if np.isnan(std_ret) or std_ret == 0:
        sharpe = np.nan
    else:
        sharpe = np.sqrt(252.0) * mean_ret / std_ret

    print("\n=== Evaluation on Final 20% of Data ===")
    print(f"Number of evaluation steps: {len(portfolio_returns)}")
    print(f"Initial Wealth: {INITIAL_WEALTH:,.2f}")
    print(f"Final Wealth:   {final_wealth:,.2f}")
    print(f"Total Return:   {100 * total_return:,.2f}%")
    print(f"Mean Daily Return: {mean_ret:.6f}")
    print(f"Std Daily Return:  {std_ret:.6f}")
    print(f"Sharpe Ratio (annualized): {sharpe:.3f}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    evaluate_model(MODEL_PATH)
