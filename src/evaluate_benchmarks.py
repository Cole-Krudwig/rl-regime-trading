# src/evaluate_benchmarks.py

import os
import numpy as np
import pandas as pd

from .TradingEnv import HJBTradingEnv
from .FeatureEngine import StochasticFeatureEngine
from .TradingStrategies import TradingStrategies


DATA_PATH = "data/spy.csv"


class BenchmarkEvaluator:
    """
    Evaluates simple benchmark portfolios on the final held-out portion of the data:
      - 100% risky asset (buy & hold)
      - Constant 60/40 risky / risk-free
      - Momentum-only (action 0)
      - Mean-reversion-only (action 1)
      - Defensive-only (action 2)
    """

    def __init__(
        self,
        data_path: str = DATA_PATH,
        train_split: float = 0.8,
        annual_rfr: float = 0.02,
        trading_days_per_year: int = 252,
    ):
        self.data_path = data_path
        self.train_split = train_split
        self.annual_rfr = annual_rfr
        self.trading_days = trading_days_per_year

        # Rebuild full data + features
        self.full_df = self._build_full_data()
        self.train_df, self.test_df = self._split_train_test(self.full_df)

        # Env constants for consistency
        self.LOOKBACK_STEPS = HJBTradingEnv.LOOKBACK_STEPS
        self.INITIAL_WEALTH = float(HJBTradingEnv.INITIAL_WEALTH)
        self.FEATURE_COLS = HJBTradingEnv.FEATURE_COLS

        # Risk-free rate per step
        self.daily_rfr = (1.0 + self.annual_rfr) ** (1.0 /
                                                     self.trading_days) - 1.0

    # ---------- Data pipeline ----------

    def _build_full_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, index_col=0)

        engine = StochasticFeatureEngine(df)
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

    # ---------- Core simulation ----------

    def _simulate_from_returns(self, returns: np.ndarray) -> dict:
        """
        Given a 1D array of per-step portfolio returns, simulate wealth and compute metrics.
        Ensures metrics are plain Python floats (not numpy scalars or arrays).
        """
        returns = np.asarray(returns, dtype=float).ravel()
        wealth = float(self.INITIAL_WEALTH)

        for r in returns:
            wealth *= (1.0 + float(r))

        final_wealth = float(wealth)
        total_return = float(final_wealth / self.INITIAL_WEALTH - 1.0)

        if len(returns) > 0:
            mean_ret = float(returns.mean())
        else:
            mean_ret = float("nan")

        if len(returns) > 1:
            std_ret = float(returns.std(ddof=1))
        else:
            std_ret = float("nan")

        if np.isnan(std_ret) or std_ret == 0.0:
            sharpe = float("nan")
        else:
            sharpe = float(np.sqrt(self.trading_days) * mean_ret / std_ret)

        return {
            "n_steps": int(len(returns)),
            "final_wealth": final_wealth,
            "total_return": total_return,
            "mean_step_return": mean_ret,
            "std_step_return": std_ret,
            "sharpe": sharpe,
        }

    def evaluate_benchmarks(self):
        df = self.test_df

        # Make sure we have enough data after lookback
        t0 = self.LOOKBACK_STEPS
        if t0 >= len(df) - 1:
            raise ValueError(
                "Test set too small for evaluation with given LOOKBACK_STEPS."
            )

        # Risky returns (consistent with env): use the 'Log_Return' column
        risky_returns = df["Log_Return"].values

        # ---- 1. 100% risky asset (buy & hold) ----
        bh_returns = risky_returns[t0 + 1:]
        bh_metrics = self._simulate_from_returns(bh_returns)

        # ---- 2. Constant 60/40 risky / risk-free ----
        rf = float(self.daily_rfr)
        sixty_forty_returns = 0.6 * risky_returns[t0 + 1:] + 0.4 * rf
        sixty_forty_metrics = self._simulate_from_returns(sixty_forty_returns)

        # ---- 3–5. Base strategies via TradingStrategies ----
        strategies = TradingStrategies(df)

        def simulate_base_strategy(action: int) -> dict:
            """
            Use TradingStrategies.calculate_strategy_returns with a fixed action.
            Robustly coerce whatever comes back into a scalar float.
            """
            step_returns = []
            for t in range(t0 + 1, len(df)):
                r = strategies.calculate_strategy_returns(t, action)

                # Robustly coerce r to scalar float
                if isinstance(r, (pd.Series, np.ndarray)):
                    r = float(np.asarray(r).ravel()[0])
                else:
                    r = float(r)

                step_returns.append(r)

            return self._simulate_from_returns(np.array(step_returns, dtype=float))

        momentum_metrics = simulate_base_strategy(action=0)
        meanrev_metrics = simulate_base_strategy(action=1)
        defensive_metrics = simulate_base_strategy(action=2)

        # ---------- Print report ----------

        print("\n========== Benchmark Performance on Final Held-Out Chunk ==========")
        header = (
            f"{'Strategy':<12} | {'Steps':>6} | {'FinalW':>12} | "
            f"{'TotRet%':>8} | {'Sharpe':>8}"
        )
        print(header)
        print("-" * len(header))

        def row(name: str, m: dict) -> str:
            return (
                f"{name:<12} | "
                f"{m['n_steps']:>6d} | "
                f"{m['final_wealth']:>12.2f} | "
                f"{m['total_return']*100:>8.2f} | "
                f"{m['sharpe']:>8.3f}"
            )

        print(row("BuyHold 100%", bh_metrics))
        print(row("60/40", sixty_forty_metrics))
        print(row("Momentum", momentum_metrics))
        print(row("MeanRevert", meanrev_metrics))
        print(row("Defensive", defensive_metrics))


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at: {DATA_PATH}")

    evaluator = BenchmarkEvaluator(DATA_PATH, train_split=0.8)
    evaluator.evaluate_benchmarks()
