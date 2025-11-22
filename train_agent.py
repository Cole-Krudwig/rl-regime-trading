# train_agent.py

import os
import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

from src.TradingEnv import HJBTradingEnv
from src.StochasticFeatureEngine import StochasticFeatureEngine


DATA_PATH = "data/spy.csv"
LOG_DIR = "logs/dqn_hjb"
MODEL_DIR = "models"


def make_trading_env(train: bool = True) -> gym.Env:
    """
    Factory function to create a fresh HJBTradingEnv instance.
    You can use different splits for train/eval if you want.
    """
    df = pd.read_csv(DATA_PATH, index_col=0)

    # Feature engineering
    engine = StochasticFeatureEngine(df)
    processed_df = engine.calculate_all_features()
    merged_df = pd.merge(
        df,
        processed_df,
        left_index=True,
        right_index=True,
        how="inner"
    )

    # Optional: train / eval split
    split_idx = int(0.8 * len(merged_df))
    if train:
        data_slice = merged_df.iloc[:split_idx].copy()
    else:
        data_slice = merged_df.iloc[split_idx:].copy()

    env = HJBTradingEnv(data=data_slice)
    env = Monitor(env)
    return env


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Vectorized env for SB3 (even if it's just 1 env)
    def _make_train_env():
        return make_trading_env(train=True)

    def _make_eval_env():
        return make_trading_env(train=False)

    train_env = DummyVecEnv([_make_train_env])
    # We skip eval_env for now to avoid shape mismatches during evaluation

    # --- Callbacks ---

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=MODEL_DIR,
        name_prefix="dqn_hjb_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    # --- DQN Model ---

    model = DQN(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=.1,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=64,
        tau=0.1,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.7,
        exploration_final_eps=0.1,
        verbose=1,
        tensorboard_log=LOG_DIR,
        # you can fix the seed if you want reproducibility
        # seed=42,
    )

    # --- Training ---
    # --- Training ---
    # --- Training ---
    total_timesteps = 10000  # tweak as needed

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback
    )

    # Save final model
    model_path = os.path.join(MODEL_DIR, "dqn_hjb_final")
    model.save(model_path)
    print(f"Saved final model to: {model_path}")

    # --- Simple evaluation rollout using train_env ---
    # Reset vectorized env (DummyVecEnv)
    obs = train_env.reset()
    done = False
    total_reward = 0.0
    final_wealth = None  # we will fill this from infos

    while not done:
        print('=' * 50)
        action, _ = model.predict(obs, deterministic=True)
        print(action)

        obs, rewards, dones, infos = train_env.step(action)
        total_reward += float(rewards[0])
        done = bool(dones[0])

        if done:
            # infos is a list of dicts, one per env (we have 1 env)
            final_wealth = infos[0].get("terminal_wealth", None)

    # Fallback: if for some reason info wasn't set, use the env attribute
    if final_wealth is None:
        monitor_env = train_env.envs[0]
        base_env = monitor_env.env
        final_wealth = base_env.current_wealth

    print("\n=== Evaluation Summary (train slice) ===")
    print(f"Final Wealth: {final_wealth:.2f}")
    print(f"Cumulative Reward: {total_reward:.9f}")

    train_env.close()


if __name__ == "__main__":
    main()
