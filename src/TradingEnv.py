import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math

from Fetch import Fetch
from StochasticFeatureEngine import StochasticFeatureEngine
from TradingStrategies import TradingStrategies


class HJBTradingEnv(gym.Env):
    """
    Gymnasium environment for HJB-Guided Stochastic Regime Meta-Control.
    The agent maximizes the change in Power Utility (HJB objective).
    """

    # --- HJB and Utility Parameters ---
    # CRRA Utility parameter (Gamma > 1 for risk aversion)
    RISK_AVERSION_GAMMA = 2.0
    INITIAL_WEALTH = 10000.0       # Starting portfolio value
    FEATURE_COLS = ['Drift_Short_Z', 'Drift_Long_Z', 'RV_Signal_Z',
                    'Vol_Trend_Z', 'MR_Residual_Z', 'Price_Velocity_Z']

    LOOKBACK_STEPS = 5

    def __init__(self, data: pd.DataFrame, max_lookback_steps=252*2):
        """
        Initializes the environment.

        Args:
            processed_data: DataFrame containing 'close', 'Log_Return', and all '_Z' features.
            max_lookback_steps: The minimum number of steps to ensure a stable observation space.
        """
        super().__init__()

        # --- Data Setup ---
        # Ensure the index is a simple integer range after dropna() to avoid Timestamp issues.
        self.data = data
        self.strategies = TradingStrategies(self.data)

        # FIX: start_step is now always 0 because we reset the index.
        self.start_step = 0
        self.max_steps = len(self.data) - 1

        # --- Action Space (3 discrete strategies) ---
        # 0: Momentum, 1: Mean Reversion, 2: Defensive
        self.action_space = spaces.Discrete(3)

        num_base_features = len(self.FEATURE_COLS) + 1

        # Total features = Base features * Lookback Steps
        total_obs_size = num_base_features * self.LOOKBACK_STEPS

        # Define boundaries: Normalized features are roughly [-5, 5], Wealth is positive
        low_bound_base = np.array(
            [-5.0] * (num_base_features - 1) + [0.0])  # Stochastic + Wealth
        high_bound_base = np.array(
            [5.0] * (num_base_features - 1) + [np.finfo(np.float64).max])

        # Stack the bounds based on the lookback steps
        low_bound = np.tile(low_bound_base, self.LOOKBACK_STEPS)
        high_bound = np.tile(high_bound_base, self.LOOKBACK_STEPS)

        self.observation_space = spaces.Box(
            low=low_bound, high=high_bound, dtype=np.float64)

    def _power_utility(self, wealth: float) -> float:
        """
        Calculates the Power Utility (CRRA) value for wealth W, the HJB objective.
        U(W) = W^(1-gamma) / (1-gamma)
        """
        gamma = self.RISK_AVERSION_GAMMA
        if wealth <= 0:
            return -1e10  # Severe penalty for ruin

        # For gamma=2, this simplifies to -1/W
        return (wealth ** (1 - gamma)) / (1 - gamma)

    def _get_obs(self):
        """Generates the current observation vector for the agent (the Filtration F_t)."""

        # 1. Stochastic features at the current step
        # We read the feature values from the current step index
        stochastic_features = self.data[self.FEATURE_COLS].iloc[self.current_step].values

        # 2. Wealth feature (normalized by the initial wealth for stability)
        normalized_wealth = self.current_wealth / self.INITIAL_WEALTH

        # Concatenate features and wealth into the final state vector
        return np.concatenate([stochastic_features, [normalized_wealth]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        """Removes the error by ensuring that the upper bound for np.random.randint() is an integer."""
        super().reset(seed=seed)

        # Start at a random point in the data *after* the initial NaN values (self.start_step)
        # We start roughly one year into the data for stable training
        year_steps = 252

        # Ensure start index is within valid range
        start_range = self.max_steps - year_steps * 2

        # FIX: start_step is now guaranteed to be integer 0.
        if start_range < self.start_step:
            self.current_step = self.start_step
        else:
            # Start randomly after the initial feature lookback period
            # Both bounds are now guaranteed to be integers (self.start_step and start_range)
            self.current_step = np.random.randint(self.start_step, start_range)

        self.current_wealth = self.INITIAL_WEALTH
        self.last_utility = self._power_utility(self.current_wealth)

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action: int):
        """
        Performs one step of the environment based on the agent's action.
        Reward is the change in utility (HJB Objective).
        """
        self.current_step += 1

        if self.current_step >= self.max_steps:
            terminated = True
            reward = 0.0

        else:
            terminated = False

            # --- 1. Calculate Portfolio Return ---
            # t is the index of the return being realized at time t
            portfolio_return = self.strategies.calculate_strategy_returns(
                self.current_step,
                action
            )

            # Update wealth
            self.current_wealth *= (1 + portfolio_return)

            # --- 2. HJB-Informed Reward Calculation (Utility Change) ---
            current_utility = self._power_utility(self.current_wealth)

            # Reward is the immediate change in Utility, driving the HJB solution.
            reward = current_utility - self.last_utility

            # Check for ruin (early termination)
            if self.current_wealth < 0.1 * self.INITIAL_WEALTH:
                terminated = True
                reward -= 100.0  # Large penalty for deep drawdown/ruin

            self.last_utility = current_utility

        # Get next observation (even if terminated, to show final state)
        observation = self._get_obs()

        # Gymnasium step returns: observation, reward, terminated, truncated, info
        return observation, reward, terminated, False, {}

# --- 3. Example Usage and Test Harness ---
