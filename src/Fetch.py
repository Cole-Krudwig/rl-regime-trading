import os
import numpy as np
import pandas as pd
import yfinance as yf


class Fetch:
    def __init__(self, data_dir: str | None = None):
        """
        Fetches historical daily data using yfinance and saves it to CSV.

        Parameters
        ----------
        data_dir : str | None
            Directory where CSVs will be stored. If None, defaults to
            the project-level 'data' directory (one level above src/).
        """
        if data_dir is None:
            root_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(root_dir, "data")

        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch(
        self,
        symbol: str,
        start: str = "1990-01-01",
        end: str | None = None,
    ) -> pd.DataFrame:
        """
        Download historical daily OHLCV data for `symbol` using yfinance,
        compute Log_Return, and save to <data_dir>/<symbol>.csv.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns:
                ['open', 'high', 'low', 'close', 'volume', 'Log_Return']
            indexed by DatetimeIndex in ascending order.
        """
        # Download from Yahoo Finance
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            raise ValueError(f"No data returned for symbol '{symbol}'")

        # --- Robust MultiIndex flattening for yfinance -----------------------
        if isinstance(df.columns, pd.MultiIndex):
            # yfinance can return e.g. ('Open','QQQ') or ('QQQ','Open').
            # We detect which level has the price fields.
            price_keys = {"open", "high", "low",
                          "close", "adj close", "volume"}
            chosen_level = None

            for level_idx in range(df.columns.nlevels):
                level_vals = {str(c).lower()
                              for c in df.columns.get_level_values(level_idx)}
                # at least 3 matches -> it's the field level
                if len(price_keys & level_vals) >= 3:
                    chosen_level = level_idx
                    break

            if chosen_level is None:
                raise ValueError(
                    f"Could not identify price field level in MultiIndex columns for {symbol}. "
                    f"Columns: {df.columns}"
                )

            df.columns = df.columns.get_level_values(chosen_level)

        # Ensure columns have no name (avoid extra header rows in CSV)
        df.columns.name = None

        # --- Standardize column names ----------------------------------------
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )

        # Sanity check that required columns are present
        required_cols = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns {missing} in yfinance data for {symbol}. "
                f"Got columns: {list(df.columns)}"
            )

        # Keep only the columns we care about and ensure float dtype
        df = df[required_cols].astype(float)

        # Ensure sorted by date ascending
        df = df.sort_index()

        # --- Compute log returns --------------------------------------------
        df["Log_Return"] = np.log(df["close"]).diff()

        # Drop the first row where Log_Return is NaN
        df = df.iloc[1:]

        # --- Save clean CSV: one header row only -----------------------------
        out_path = os.path.join(self.data_dir, f"{symbol}.csv")
        df.to_csv(out_path)

        return df
