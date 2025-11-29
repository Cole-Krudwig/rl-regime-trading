import numpy as np
import pandas as pd


class TradingStrategies:
    """
    Defines the discrete set of trading strategies (Actions) available to the 
    RL Meta-Controller. Each method returns:
        (weight_risky, weight_cash)
    where weight_risky + weight_cash = 1.
    """

    def __init__(self, data: pd.DataFrame, annual_rfr: float = 0.02, trading_days: int = 252):
        """
        Initializes the strategies class with the processed data.
        The data must include 'close', 'Log_Return', and 'MR_Residual' from feature_engine.

        annual_rfr: annualized risk-free rate (e.g. 0.02 for 2%).
        trading_days: number of trading days per year.
        """
        self.data = data

        # Strategy constants
        self.CASH_WEIGHT = 0.0
        self.FULL_LONG_WEIGHT = 1.0
        self.SMALL_SHORT_WEIGHT = -0.5  # Short up to 50% for defense/bearish view

        # Risk-free parameters
        self.annual_rfr = annual_rfr
        self.trading_days = trading_days
        # Convert annual RFR to per-period (daily) rate
        self.daily_rfr = (1 + self.annual_rfr) ** (1 / self.trading_days) - 1

    def momentum_strategy(self, t: int) -> tuple[float, float]:
        """
        Action 0: Momentum/Trend Following (Best for Bull Regime).
        Rule: Full allocation to the risky asset.

        Returns:
            (weight_risky, weight_cash)
        """
        weight_risky = self.FULL_LONG_WEIGHT
        weight_cash = 1 - weight_risky
        return weight_risky, weight_cash

    def mean_reversion_strategy(self, t: int) -> tuple[float, float]:
        """
        Action 1: Mean Reversion (Best for Flat/Choppy Regime).

        - If price is HIGH relative to its mean (positive residual), SELL (short)
        - If price is LOW relative to its mean (negative residual), BUY (long)

        Returns:
            (weight_risky, weight_cash)
        """
        try:
            # Use previous day's residual for non-anticipative behavior
            residual = self.data['MR_Residual_Z'].iloc[t - 1]
        except (IndexError, KeyError):
            # First day or missing column: stay fully in cash
            return 0.0, 1.0

        # Opposite the residual: high => short, low => long
        weight_risky = np.clip(-0.5 * residual, -1.0, 1.0)
        weight_cash = 1 - weight_risky
        return weight_risky, weight_cash

    def defensive_strategy(self, t: int) -> tuple[float, float]:
        """
        Action 2: Defensive/Anti-Momentum (Best for Bear/High-Volatility Regime).
        Rule: Small short position + the rest in risk-free.
        """
        weight_risky = self.SMALL_SHORT_WEIGHT
        weight_cash = 1 - weight_risky
        return weight_risky, weight_cash

    def calculate_strategy_returns(self, t: int, action: int) -> float:
        """
        Calculates the portfolio return for a given day (t) and chosen action (strategy).

        Args:
            t (int): The current index of the day in the data.
            action (int): The discrete action chosen by the RL agent (0, 1, or 2).

        Returns:
            float: The portfolio *arithmetic* return for the day.
        """

        # Map discrete action index to the strategy weight function
        if action == 0:
            weight_risky, weight_cash = self.momentum_strategy(t)
        elif action == 1:
            weight_risky, weight_cash = self.mean_reversion_strategy(t)
        elif action == 2:
            weight_risky, weight_cash = self.defensive_strategy(t)
        else:
            raise ValueError("Invalid action index (must be 0, 1, or 2).")

        # Risky asset (log) return for day t
        risky_return = self.data['Log_Return'].iloc[t]

        # print(f"Weight: {weight_risky}")
        # print(f"Weight Cash: {weight_cash}")

        # Risk-free per-period return
        risk_free_return = self.daily_rfr

        # Portfolio arithmetic return for this step
        portfolio_return = weight_risky * risky_return + weight_cash * risk_free_return
        return portfolio_return
