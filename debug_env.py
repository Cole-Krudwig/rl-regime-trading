import pandas as pd
import numpy as np
from src.TradingEnv import HJBTradingEnv
from src.StochasticFeatureEngine import StochasticFeatureEngine

df = pd.read_csv("data/spy.csv", index_col=0)
engine = StochasticFeatureEngine(df)
processed_df = engine.calculate_all_features()
merged_df = pd.merge(df, processed_df, left_index=True,
                     right_index=True, how="inner")

env = HJBTradingEnv(data=merged_df)

obs, info = env.reset()
total_r = 0.0

for t in range(500):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    total_r += r
    if term or trunc:
        break

print("Random policy final wealth:", env.current_wealth)
print("Random policy cumulative reward:", total_r)
