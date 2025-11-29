import numpy as np
import pandas as pd


class StochasticFeatureEngine:
    """
    Computes all necessary stochastic features for the RL agent's state space
    from raw daily financial data. 

    Features include proxies for SDE drift (momentum) and volatility (quadratic variation).
    """

    # Define standard look-back windows for feature calculations
    WINDOWS = {
        'SHORT_TERM_MOMENTUM': 5,     # Proxy for instantaneous drift (mu)
        'LONG_TERM_MOMENTUM': 60,     # Contextual drift
        'VOLATILITY': 30,             # Window for Quadratic Variation/Realized Volatility
        'MEAN_REVERSION': 20          # Window for mean reversion residual
    }

    # List of features that will be used in the final RL state vector
    FEATURE_COLUMNS = []

    def __init__(self, data: pd.DataFrame):
        """Initializes the engine with the raw data."""
        self.data = data.copy()

    def calculate_all_features(self) -> pd.DataFrame:
        """Runs all feature generation and normalization steps."""

        self.data = self._calculate_log_returns(self.data)

        self._add_directional_features()
        self._add_volatility_features()
        self._add_mean_reversion_features()
        self._add_log_returns()

        # Drop rows with NaN values resulting from rolling window calculations
        self.data.dropna(inplace=True)

        # Normalize only the features used in the final state vector
        self._normalize_features()

        # Select only the features needed for the RL state
        return self.data[self.FEATURE_COLUMNS]

    def _calculate_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates the foundational log returns."""
        # Log return is the instantaneous return used in SDEs
        df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))

        # Squared Log Returns (Daily Quadratic Variation)
        df['Log_Return_Sq'] = df['Log_Return']**2
        return df

    def _add_directional_features(self):
        """
        Adds proxies for the SDE Drift (mu) and Momentum.
        """
        # --- Feature 1: Short-Term Momentum (Instantaneous Drift Proxy) ---
        win_s = self.WINDOWS['SHORT_TERM_MOMENTUM']
        col_s = 'Drift_Short'
        self.data[col_s] = self.data['Log_Return'].rolling(
            window=win_s).mean() * 252
        self.FEATURE_COLUMNS.append(col_s + '_Z')

        # --- Feature 2: Long-Term Drift (Contextual Momentum) ---
        win_l = self.WINDOWS['LONG_TERM_MOMENTUM']
        col_l = 'Drift_Long'
        self.data[col_l] = self.data['Log_Return'].rolling(
            window=win_l).mean() * 252
        self.FEATURE_COLUMNS.append(col_l + '_Z')

    def _add_volatility_features(self):
        """
        Adds proxies for SDE Volatility (sigma) using Quadratic Variation (QV).
        """
        win_v = self.WINDOWS['VOLATILITY']

        # --- Feature 3: Realized Volatility (Proxy for sigma) ---
        # QV is the sum of squared log returns. RV is the square root of QV.
        # Annualized Realized Volatility = sqrt( sum(Log_Return_Sq) * (252 / window) )
        col_rv = 'RV_Signal'
        daily_vol_factor = np.sqrt(252 / win_v)

        self.data[col_rv] = np.sqrt(self.data['Log_Return_Sq'].rolling(
            window=win_v).sum()) * daily_vol_factor
        self.FEATURE_COLUMNS.append(col_rv + '_Z')

        # --- Feature 4: Volatility Trend/Acceleration ---
        # Measures the change in volatility to detect entry/exit from high-vol regime
        col_vt = 'Vol_Trend'
        # Compare current RV with RV 20 days ago
        self.data[col_vt] = self.data[col_rv].diff(periods=20)
        self.FEATURE_COLUMNS.append(col_vt + '_Z')

    def _add_mean_reversion_features(self):
        """
        Adds features essential for distinguishing the Flat/Mean Reversion regime.
        """
        win_mr = self.WINDOWS['MEAN_REVERSION']

        # --- Feature 5: Mean Reversion Residual ---
        # How far the price is from its mean, normalized by its current volatility.
        # Price is typically non-stationary, so we use the Log of the price ratio (return)

        # Calculate a rolling average price
        self.data['MA_Price'] = self.data['close'].rolling(
            window=win_mr).mean()

        # Residual: log(Current Price / MA Price) -> measures percentage deviation
        col_res = 'MR_Residual'
        self.data[col_res] = np.log(self.data['close'] / self.data['MA_Price'])
        self.FEATURE_COLUMNS.append(col_res + '_Z')

        # --- Feature 6: Price Velocity (Used to filter chop vs true trend) ---
        # Simple percentage change over a medium period.
        col_vel = 'Price_Velocity'
        self.data[col_vel] = self.data['close'].pct_change(periods=win_mr)
        self.FEATURE_COLUMNS.append(col_vel + '_Z')

    def _add_log_returns(self):
        self.data['Log_Return'] = np.log(
            self.data['close'] / self.data['close'].shift(1))

    def _normalize_features(self):
        """
        Normalizes all final feature columns using Z-score standardization.
        This is necessary for stable neural network training.
        """
        for col in self.FEATURE_COLUMNS:
            # The column exists without the '_Z' suffix
            base_col = col.replace('_Z', '')

            mean = self.data[base_col].mean()
            std = self.data[base_col].std()

            # Standardization (Z-score): (X - mu) / sigma
            # This scales all features to have a mean of 0 and a standard deviation of 1.
            self.data[col] = (self.data[base_col] - mean) / std


# --- Example Usage (Assuming data is loaded into a DataFrame 'raw_df') ---

# The user is assumed to have loaded this raw data:
# raw_data = {
#     'open': [682.73, 687.05, 688.72, 683.90, 685.04],
#     'high': [685.54, 688.905, 689.70, 685.94, 685.08],
#     'low': [682.115, 684.83, 682.87, 679.83, 679.24],
#     'close': [685.24, 687.06, 687.39, 679.83, 682.06],
#     'volume': [63339790, 61738073, 85362207, 76335751, 87164122]
# }
# index = pd.to_datetime(['2025-10-27', '2025-10-28', '2025-10-29', '2025-10-30', '2025-10-31'])
# raw_df = pd.DataFrame(raw_data, index=index)
#
# engine = StochasticFeatureEngine(raw_df)
# processed_df = engine.calculate_all_features()
# print(processed_df)
