# evaluate_model.py

import os
import numpy as np
import pandas as pd

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.base_class import BaseAlgorithm

from .TradingEnv import TradingEnv
from .FeatureEngine import FeatureEngine
from .TradingStrategies import TradingStrategies


# Paths
DATA_PATH = "data/spy.csv"

# These should match the paths used in your Train class
DQN_MODEL_PATH = "models/dqn/dqn_hjb_final.zip"
PPO_MODEL_PATH = "models/ppo/ppo_hjb_final.zip"
A2C_MODEL_PATH = "models/a2c/a2c_hjb_final.zip"


class ModelEvaluator:
    """
    Evaluates one or more trained RL models on the final 20% of the data.
    For each model, it runs a single pass through the test slice and computes:
      - number of steps
      - final wealth
      - total return
      - mean step return
      - std step return
      - annualized Sharpe ratio
    """

    def __init__(self, data_path: str = DATA_PATH, train_split: float = 0.8):
        self.data_path = data_path
        self.train_split = train_split

        # Build full data + features once
        self.full_df = self._build_full_data()
        self.train_df, self.test_df = self._split_train_test(self.full_df)

        # Pull constants from env class
        self.LOOKBACK_STEPS = TradingEnv.LOOKBACK_STEPS
        self.FEATURE_COLS = TradingEnv.FEATURE_COLS
        self.INITIAL_WEALTH = TradingEnv.INITIAL_WEALTH

    # ---------- Data pipeline ----------

    def _build_full_data(self) -> pd.DataFrame:
        """
        Rebuilds the data + feature pipeline (same as in training).
        """
        df = pd.read_csv(self.data_path, index_col=0)

        engine = FeatureEngine(df)
        processed_df = engine.calculate_all_features()

        merged_df = pd.merge(
            df,
            processed_df,
            left_index=True,
            right_index=True,
            how="inner",
        ).reset_index(drop=True)

        return merged_df

    def _split_train_test(self, df: pd.DataFrame):
        split_idx = int(self.train_split * len(df))
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
        return train_df, test_df

    # ---------- Core evaluation logic ----------

    def _make_base_obs(self, df: pd.DataFrame, t_idx: int, wealth: float) -> np.ndarray:
        """
        Mimics TradingEnv._get_base_obs:
        [ stochastic Z-features at t, normalized wealth ]
        """
        features = df.loc[t_idx, self.FEATURE_COLS].values.astype(np.float32)
        norm_wealth = np.float32(wealth / self.INITIAL_WEALTH)
        base = np.concatenate([features, [norm_wealth]]).astype(np.float32)
        return base

    def _evaluate_single_model(self, model: BaseAlgorithm, df: pd.DataFrame) -> dict:
        """
        Runs a single model through the test slice once and returns metrics dict.

        In addition to summary statistics, this function records:
          - actions: sequence of chosen actions (for regime-usage plots)
          - wealth_path: wealth at each step (for equity curves)
          - action_counts: count of each action {0, 1, 2}
        """
        strategies = TradingStrategies(df)

        # Start wealth
        wealth = self.INITIAL_WEALTH
        portfolio_returns: list[float] = []
        actions: list[int] = []
        wealth_path: list[float] = [wealth]  # include initial wealth at t0

        # Burn-in: mimic env.reset() behavior
        t0 = self.LOOKBACK_STEPS
        if t0 >= len(df) - 1:
            raise ValueError(
                "Test set too small for evaluation with given LOOKBACK_STEPS."
            )

        base_obs = self._make_base_obs(df, t0, wealth)
        obs_history = [base_obs for _ in range(self.LOOKBACK_STEPS)]
        current_t = t0

        while True:
            # Build stacked observation [x_{t-L+1}, ..., x_t]
            obs = np.concatenate(obs_history, axis=0).astype(np.float32)

            # Predict action
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            actions.append(action)

            next_t = current_t + 1
            if next_t >= len(df):
                break  # no more data

            # Compute portfolio return and update wealth
            portfolio_return = strategies.calculate_strategy_returns(
                next_t, action)
            portfolio_returns.append(portfolio_return)

            wealth *= (1.0 + portfolio_return)
            wealth_path.append(wealth)

            # Build next obs
            next_base_obs = self._make_base_obs(df, next_t, wealth)
            obs_history.pop(0)
            obs_history.append(next_base_obs)

            current_t = next_t

        # ----- Metrics -----
        final_wealth = wealth
        total_return = final_wealth / self.INITIAL_WEALTH - 1.0

        returns_arr = np.array(portfolio_returns, dtype=np.float64)
        mean_ret = returns_arr.mean() if len(returns_arr) > 0 else np.nan
        std_ret = returns_arr.std(ddof=1) if len(returns_arr) > 1 else np.nan

        # Annualized Sharpe assuming ~252 trading days
        if np.isnan(std_ret) or std_ret == 0:
            sharpe = np.nan
        else:
            sharpe = np.sqrt(252.0) * mean_ret / std_ret

        # Count how many times each action was chosen (0, 1, 2)
        action_counts = {0: 0, 1: 0, 2: 0}
        for a in actions:
            if a in action_counts:
                action_counts[a] += 1

        return {
            "n_steps": len(portfolio_returns),
            "initial_wealth": self.INITIAL_WEALTH,
            "final_wealth": final_wealth,
            "total_return": total_return,
            "mean_step_return": mean_ret,
            "std_step_return": std_ret,
            "sharpe": sharpe,
            # for plots
            "actions": np.array(actions, dtype=int),
            "wealth_path": np.array(wealth_path, dtype=float),
            "action_counts": action_counts,
        }

    # ---------- Public API ----------

    def evaluate_all(self) -> None:
        """
        Loads DQN, PPO, and A2C models (if present), evaluates each on the test slice,
        and prints a comparison report.

        Additionally, it stores:
          - action frequency data
          - equity curves
        as CSV files for external plotting (e.g., in LaTeX).
        """
        models_info = {
            "DQN": (DQN, DQN_MODEL_PATH),
            "PPO": (PPO, PPO_MODEL_PATH),
            "A2C": (A2C, A2C_MODEL_PATH),
        }

        results = {}

        # directory to store plot data
        plot_dir = os.path.join(os.path.dirname(self.data_path), "plot_data")
        os.makedirs(plot_dir, exist_ok=True)

        for name, (cls, path) in models_info.items():
            if not os.path.exists(path):
                print(f"[{name}] Model file not found at: {path} (skipping)")
                continue

            print(f"\n=== Evaluating {name} on final 20% of data ===")
            model = cls.load(path)
            metrics = self._evaluate_single_model(model, self.test_df)
            results[name] = metrics

            print(f"  Steps:            {metrics['n_steps']}")
            print(f"  Initial Wealth:   {metrics['initial_wealth']:.2f}")
            print(f"  Final Wealth:     {metrics['final_wealth']:.2f}")
            print(f"  Total Return:     {metrics['total_return']*100:.2f}%")
            print(f"  Mean Step Return: {metrics['mean_step_return']:.6f}")
            print(f"  Std Step Return:  {metrics['std_step_return']:.6f}")
            print(f"  Sharpe (annual.): {metrics['sharpe']:.3f}")

            # --- NEW: Save data for plots ---

            # 1) Equity curve: step index vs. wealth
            wealth_path = metrics["wealth_path"]
            equity_df = pd.DataFrame(
                {
                    "step": np.arange(len(wealth_path)),
                    "wealth": wealth_path,
                }
            )
            equity_path = os.path.join(plot_dir, f"{name}_equity_curve.csv")
            equity_df.to_csv(equity_path, index=False)

            # 2) Action sequence: step index vs. action
            actions = metrics["actions"]
            actions_df = pd.DataFrame(
                {
                    "step": np.arange(len(actions)),
                    "action": actions,
                }
            )
            actions_path = os.path.join(plot_dir, f"{name}_actions.csv")
            actions_df.to_csv(actions_path, index=False)

            # 3) Action counts summary: one row per model
            ac = metrics["action_counts"]
            # We'll accumulate these and write once after the loop
            metrics["action_counts_row"] = {
                "model": name,
                "action_0": ac.get(0, 0),
                "action_1": ac.get(1, 0),
                "action_2": ac.get(2, 0),
            }

        # --------- Small summary report ---------
        if not results:
            print("\nNo models were evaluated.")
            return

        print("\n========== Summary: Test Performance on Final 20% ==========")
        header = (
            f"{'Model':<6} | {'Steps':>6} | {'FinalW':>12} | "
            f"{'TotRet%':>8} | {'Sharpe':>8}"
        )
        print(header)
        print("-" * len(header))

        action_rows = []

        for name, m in results.items():
            print(
                f"{name:<6} | "
                f"{m['n_steps']:>6d} | "
                f"{m['final_wealth']:>12.2f} | "
                f"{m['total_return']*100:>8.2f} | "
                f"{m['sharpe']:>8.3f}"
            )
            if "action_counts_row" in m:
                action_rows.append(m["action_counts_row"])

        # Save a single CSV with action counts for all models (for bar charts)
        if action_rows:
            action_counts_df = pd.DataFrame(action_rows)
            action_counts_path = os.path.join(
                plot_dir, "action_counts_summary.csv")
            action_counts_df.to_csv(action_counts_path, index=False)
            print(
                f"\n[INFO] Saved action counts summary to: {action_counts_path}")


'''
if __name__ == "__main__":
    evaluator = ModelEvaluator(DATA_PATH, train_split=0.8)
    evaluator.evaluate_all()
'''
