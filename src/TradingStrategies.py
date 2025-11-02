import numpy as np
import pandas as pd


class TradingStrategies:
    """
    Defines the discrete set of trading strategies (Actions) available to the 
    RL Meta-Controller. Each method returns the portfolio weight (pi_t) 
    in the risky asset for the current day based on fixed, simple rules.

    The Meta-Controller's action A_t maps to one of these three methods.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the strategies class with the processed data.
        The data must include 'close', 'Log_Return', and 'MR_Residual' from feature_engine.
        """
        self.data = data

        # Define constants for simple strategy logic
        self.CASH_WEIGHT = 0.0
        self.FULL_LONG_WEIGHT = 1.0
        self.SMALL_SHORT_WEIGHT = -0.5  # Short up to 50% for defense/bearish view

    def momentum_strategy(self, t: int) -> float:
        """
        Action 1: Momentum/Trend Following (Best for Bull Regime).
        Rule: Full allocation to the risky asset.
        Returns: Portfolio weight (pi_t) in the risky asset.
        """
        # We assume the decision to enter this strategy means going fully long.
        return self.FULL_LONG_WEIGHT

    def mean_reversion_strategy(self, t: int) -> float:
        """
        Action 2: Mean Reversion (Best for Flat/Choppy Regime).
        Rule: Bet against the short-term deviation from the mean (MR_Residual).

        - If price is HIGH relative to its mean (positive residual), SELL (short)
        - If price is LOW relative to its mean (negative residual), BUY (long)

        The weight magnitude is scaled by the residual magnitude, capped at +/- 1.0.
        """
        # The 'MR_Residual_Z' is a standardized measure of how far price is from its 20-day mean.
        # We use the previous day's residual (t-1) for a non-anticipative policy.
        try:
            residual = self.data['MR_Residual_Z'].iloc[t-1]
        except IndexError:
            # Default to cash if index is invalid (e.g., first day)
            return self.CASH_WEIGHT

        # The weight is opposite the residual, capped at +/- 1.0.
        # The scaling factor (e.g., -1.0) determines the aggressiveness.
        weight = np.clip(-residual * 0.5, -1.0, 1.0)

        return weight

    def defensive_strategy(self, t: int) -> float:
        """
        Action 3: Defensive/Anti-Momentum (Best for Bear/High-Volatility Regime).
        Rule: Hold cash or take a small short position to benefit from declines/protect against risk.

        For simplicity, we use a fixed negative weight.
        """
        # Use a small short position (50% short) to bet against the trend,
        # but the primary goal is risk reduction compared to A1 and A2.
        return self.SMALL_SHORT_WEIGHT

    def calculate_strategy_returns(self, t: int, action: int) -> float:
        """
        Calculates the portfolio return for a given day (t) and chosen action (strategy).

        Args:
            t (int): The current index of the day in the data.
            action (int): The discrete action chosen by the RL agent (0, 1, or 2).

        Returns:
            float: The portfolio return for the day.
        """

        # Map discrete action index to the strategy weight function
        if action == 0:
            weight = self.momentum_strategy(t)
        elif action == 1:
            weight = self.mean_reversion_strategy(t)
        elif action == 2:
            weight = self.defensive_strategy(t)
        else:
            raise ValueError("Invalid action index.")

        # Portfolio Return = (Weight in Risky Asset * Risky Asset Return) + (Weight in Cash * Risk-Free Rate)
        # We assume the risk-free rate is 0 over a single day (dt).

        risky_return = self.data['Log_Return'].iloc[t]

        # Portfolio return is simplified to pi_t * Risky_Return
        return weight * risky_return
