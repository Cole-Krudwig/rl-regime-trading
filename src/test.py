import pandas as pd
import numpy as np

import TradingEnv
from TradingEnv import HJBTradingEnv
from Fetch import Fetch
from StochasticFeatureEngine import StochasticFeatureEngine
from TradingStrategies import TradingStrategies

if __name__ == '__main__':

    fetcher = Fetch()
    df = fetcher.fetch("SPY")
    engine = StochasticFeatureEngine(df)
    processed_df = engine.calculate_all_features()
    # print(processed_df.head())

    merged_df = pd.merge(df, processed_df, left_index=True,
                         right_index=True, how="inner")

    print(merged_df.head())
    print(len(merged_df))

    env = HJBTradingEnv(data=merged_df)

    print("--- HJB Trading Environment Test ---")

    # Test reset
    obs, info = env.reset()
    print(obs, info)
    print(f"1. Environment Reset. Starting Index: {env.current_step}")
    print(
        f"   Initial Observation (Drift, RV, MR, Wealth): {np.round(obs, 4)}")
    print(f"   Initial Utility (V_0): {round(env.last_utility, 4)}")

    # Define a simple sequence of actions: 5x Momentum (0), 5x Defensive (2)
    test_actions = [0] * 5 + [2] * 5
    total_reward = 0

    print("\n2. Simulating 10 Steps (Momentum then Defensive):")

    for i, action in enumerate(test_actions):

        # Get price information for printout
        current_date = env.data.index[env.current_step + 1].date()

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Log details
        w_norm = round(obs[-1], 6)
        w_final = round(w_norm * env.INITIAL_WEALTH, 2)

        print(f"   [{current_date}] Action {action}: Wealth={w_final} | Utility Change={reward:+.6f} | Terminated={terminated}")

        if terminated:
            break

    print("\n3. Simulation Summary:")
    print(f"   Total Cumulative HJB Reward: {total_reward:+.4f}")
    print(f"   Final Portfolio Value: {w_final}")
