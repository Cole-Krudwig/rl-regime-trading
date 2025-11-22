import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math

from src.Fetch import Fetch
from src.StochasticFeatureEngine import StochasticFeatureEngine
from src.TradingStrategies import TradingStrategies


class HJBTradingEnv(gym.Env):
    """
    Gymnasium environment for HJB-Guided Stochastic Regime Meta-Control.
    The agent maximizes the change in Power Utility (HJB objective).
    """

    # --- HJB and Utility Parameters ---
    # CRRA Utility parameter (Gamma > 1 for risk aversion)
    RISK_AVERSION_GAMMA = 2
    INITIAL_WEALTH = 10000.0       # Starting portfolio value
    REWARD_SCALE = 1e5
    FEATURE_COLS = ['Drift_Short_Z', 'Drift_Long_Z', 'RV_Signal_Z',
                    'Vol_Trend_Z', 'MR_Residual_Z', 'Price_Velocity_Z']

    LOOKBACK_STEPS = 14

    def __init__(self, data: pd.DataFrame, max_lookback_steps=252*2):
        """
        Initializes the environment.

        Args:
            data: DataFrame containing 'close', 'Log_Return', and all '_Z' features.
            max_lookback_steps: The minimum number of steps to ensure a stable observation space.
        """
        super().__init__()

        # --- Data Setup ---
        self.data = data
        self.strategies = TradingStrategies(self.data)

        # Start index and max steps
        self.start_step = 0
        self.max_steps = len(self.data) - 1

        # --- Action Space (3 discrete strategies) ---
        # 0: Momentum, 1: Mean Reversion, 2: Defensive
        self.action_space = spaces.Discrete(3)

        # --- Observation Space (lookback x base features) ---
        # base features = stochastic features + wealth
        self.num_base_features = len(self.FEATURE_COLS) + 1
        total_obs_size = self.num_base_features * self.LOOKBACK_STEPS

        # Normalized features roughly [-5, 5], wealth >= 0
        low_bound_base = np.array(
            [-5.0] * (self.num_base_features - 1) + [0.0], dtype=np.float32
        )
        high_bound_base = np.array(
            [5.0] * (self.num_base_features - 1) + [np.finfo(np.float32).max],
            dtype=np.float32
        )

        low_bound = np.tile(low_bound_base, self.LOOKBACK_STEPS)
        high_bound = np.tile(high_bound_base, self.LOOKBACK_STEPS)

        self.observation_space = spaces.Box(
            low=low_bound,
            high=high_bound,
            dtype=np.float32
        )

        # Will hold the last LOOKBACK_STEPS base observations
        self.obs_history = None

        # These get set in reset()
        self.current_step = None
        self.current_wealth = None
        self.last_utility = None

    # ---------- Utility & Observation Helpers ----------

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

    def _get_base_obs(self) -> np.ndarray:
        """
        Base observation at current time step:
        [ stochastic Z-features..., normalized wealth ]
        Shape: (num_base_features,)
        """
        stochastic_features = self.data[self.FEATURE_COLS].iloc[
            self.current_step
        ].values.astype(np.float32)

        normalized_wealth = np.float32(
            self.current_wealth / self.INITIAL_WEALTH)

        return np.concatenate([stochastic_features, [normalized_wealth]]).astype(
            np.float32
        )

    def _get_obs(self) -> np.ndarray:
        """
        Returns the stacked lookback observation:

            [x_{t-L+1}, ..., x_t]

        where each x_k is `num_base_features` long.

        Shape: (LOOKBACK_STEPS * num_base_features,)
        """

        # If history hasn't been initialized yet (shouldn't happen after reset),
        # initialize by repeating the current base obs.
        if self.obs_history is None:
            base_obs = self._get_base_obs()
            self.obs_history = [base_obs for _ in range(self.LOOKBACK_STEPS)]

        stacked = np.concatenate(self.obs_history, axis=0).astype(np.float32)
        return stacked

    # ---------- Gym API ----------

    def reset(self, seed=None, options=None):
        """
        Reset environment and initialize lookback history.
        """
        super().reset(seed=seed)

        # Start at a random point in the data *after* some burn-in
        year_steps = 252
        start_range = self.max_steps - year_steps * 2

        if start_range < self.start_step:
            self.current_step = self.start_step
        else:
            self.current_step = np.random.randint(self.start_step, start_range)

        self.current_wealth = self.INITIAL_WEALTH
        self.last_utility = self._power_utility(self.current_wealth)

        # Initialize history with the same base obs repeated LOOKBACK_STEPS times
        base_obs = self._get_base_obs()
        self.obs_history = [base_obs for _ in range(self.LOOKBACK_STEPS)]

        observation = self._get_obs()
        info = {}

        return observation, info

    def step(self, action: int):
        """
        Performs one step of the environment based on the agent's action.
        Reward is the change in utility (HJB Objective).
        """
        info = {}

        # Move to next time step
        self.current_step += 1

        if self.current_step >= self.max_steps:
            # We're at or beyond the last valid time index
            terminated = True
            reward = 0.0

            # Do NOT update wealth; just return final stacked obs
            # But record the final wealth
            info["terminal_wealth"] = self.current_wealth

        else:
            terminated = False

            # --- 1. Calculate Portfolio Return ---
            portfolio_return = self.strategies.calculate_strategy_returns(
                self.current_step,
                action
            )
            print(f"Portfolio Return: {portfolio_return}")
            # Update wealth
            self.current_wealth *= (1 + portfolio_return)
            print(f"Wealth: {self.current_wealth}")

            # --- 2. HJB-Informed Reward Calculation (Utility Change) ---
            current_utility = self._power_utility(self.current_wealth)
            reward = self.REWARD_SCALE * (current_utility - self.last_utility)

            # Check for ruin (early termination)
            if self.current_wealth < 0.1 * self.INITIAL_WEALTH:
                terminated = True
                reward -= 100.0  # Large penalty for deep drawdown/ruin
                info["terminal_wealth"] = self.current_wealth

            self.last_utility = current_utility

        # Update observation history with latest base obs
        # (even if terminated, as long as current_step is within bounds)
        if self.current_step <= self.max_steps:
            base_obs = self._get_base_obs()
            # Pop oldest, append newest
            self.obs_history.pop(0)
            self.obs_history.append(base_obs)

        observation = self._get_obs()
        print(f"Step obs: {observation}")

        # Gymnasium step returns: observation, reward, terminated, truncated, info
        return observation, reward, terminated, False, info
