from src.TradingEnv import HJBTradingEnv
from src.StochasticFeatureEngine import StochasticFeatureEngine

import pandas as pd

df = pd.read_csv(
    '/Users/colekrudwig/Programming/rl-regime-trading/data/spy.csv', index_col=0)
engine = StochasticFeatureEngine(df)
processed_df = engine.calculate_all_features()
# print(processed_df.head())

merged_df = pd.merge(df, processed_df, left_index=True,
                     right_index=True, how="inner")

env = HJBTradingEnv(data=merged_df)
obs, info = env.reset()
print(obs.shape)  # (35,)
print(env.observation_space.shape)  # (35,)
