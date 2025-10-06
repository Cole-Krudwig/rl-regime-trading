import pandas as pd
import numpy as np


class Process:
    def __init__(self):
        pass

    def add_log_returns(self, df, price_col='close'):
        px = df[price_col].dropna()
        r = np.log(px).diff()
        out = df.copy()
        out["r"] = r
        return out.dropna(subset=["r"])

    def add_drift_qv(self, df, r_col="r", W_mu=63, W_V=63):
        g = df.copy()
        g["mu_roll"] = g[r_col].rolling(W_mu, min_periods=W_mu).mean()
        g["qv_roll"] = (g[r_col]**2).rolling(W_V, min_periods=W_V).sum()
        # Annualize
        g["mu_ann"] = 252 * g["mu_roll"]
        g["sigma_ann"] = np.sqrt((g["qv_roll"] / W_V) * 252)
        return g.dropna(subset=["mu_ann", "sigma_ann"])

    def rolling_quantile(self, s: pd.Series, L=252, q=0.70):
        q_series = s.rolling(L, min_periods=L).quantile(q)
        return q_series.shift(1)  # avoid look-ahead

    def label_regimes(self, df, mu_col="mu_ann", sig_col="sigma_ann", L=252, high_vol_q=0.70):
        z = df.copy()
        # compute trailing (lagged) high-vol threshold from sigma series
        hv_thresh = self.rolling_quantile(z[sig_col], L=L, q=high_vol_q)

        regime = pd.Series("MR", index=z.index)  # default MR

        # Defensive (high vol)
        high_vol = z[sig_col] > hv_thresh
        regime[high_vol] = "DEF"

        # Momentum if positive drift and not high vol
        regime[(~high_vol) & (z[mu_col] > 0)] = "MOM"

        z["regime"] = regime
        return z.dropna(subset=["regime"])

    def preprocess(self, df):
        # or 'adj_close' if you have it
        r = self.add_log_returns(df, price_col="close")
        g = self.add_drift_qv(r, r_col="r", W_mu=63, W_V=63)
        z = self.label_regimes(
            g, mu_col="mu_ann", sig_col="sigma_ann", L=252, high_vol_q=0.70)
        return z
