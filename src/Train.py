# train_agent.py

import os
import numpy as np
import pandas as pd
import gymnasium as gym

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from .TradingEnv import TradingEnv
from .FeatureEngine import FeatureEngine


DATA_PATH = "data/spy.csv"
LOG_DIR = "logs"
MODEL_DIR = "models"


class Train:
    def __init__(self, data_path: str = DATA_PATH, train_split: float = 0.8):
        """
        Initialize the training pipeline:
        - Load raw data
        - Run feature engineering
        - Do a train/test split
        """
        self.data_path = data_path
        self.train_split = train_split

        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(MODEL_DIR, exist_ok=True)

        # --- Load and preprocess data once ---
        df = pd.read_csv(self.data_path, index_col=0)

        engine = FeatureEngine(df)
        processed_df = engine.calculate_all_features()

        merged_df = pd.merge(
            df,
            processed_df,
            left_index=True,
            right_index=True,
            how="inner"
        ).reset_index(drop=True)

        self.merged_df = merged_df
        self.split_idx = int(self.train_split * len(self.merged_df))

    # ------------- Env helpers -------------

    def _get_data_slice(self, train: bool = True) -> pd.DataFrame:
        if train:
            return self.merged_df.iloc[:self.split_idx].copy()
        else:
            return self.merged_df.iloc[self.split_idx:].copy()

    def make_trading_env(self, train: bool = True) -> gym.Env:
        """
        Create a single TradingEnv with train or test slice of data,
        wrapped in a Monitor.
        """
        data_slice = self._get_data_slice(train=train)
        env = TradingEnv(data=data_slice)
        env = Monitor(env)
        return env

    def _make_vec_env(self, train: bool = True) -> DummyVecEnv:
        """
        Wrap the environment in a DummyVecEnv for use with SB3.
        """
        def _init():
            return self.make_trading_env(train=train)

        return DummyVecEnv([_init])

    # ------------- Simple evaluation -------------

    @staticmethod
    def _simple_rollout(model, vec_env: DummyVecEnv, algo_name: str, max_steps: int = 10_000):
        """
        Run one episode on the given vectorized env and print
        final wealth + cumulative reward.
        """
        obs = vec_env.reset()
        done = False
        total_reward = 0.0
        final_wealth = None
        step_count = 0

        while not done and step_count < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)

            total_reward += float(rewards[0])
            done = bool(dones[0])
            step_count += 1

            if done:
                final_wealth = infos[0].get("terminal_wealth", None)

        if final_wealth is None:
            monitor_env = vec_env.envs[0]
            base_env = monitor_env.env
            final_wealth = base_env.current_wealth

        print(f"\n=== {algo_name} Evaluation Summary (train slice) ===")
        print(f"Steps:         {step_count}")
        print(f"Final Wealth:  {final_wealth:.2f}")
        print(f"Cumulative R:  {total_reward:.9f}")

    # ------------- Training for all models -------------

    def train_all(self, total_timesteps: int = 10_000):
        """
        Train DQN, PPO, and A2C on the same training environment.
        Each model is saved under models/<algo_name>/.
        """
        # Shared envs
        train_env = self._make_vec_env(train=True)

        # --------- DQN ---------
        dqn_model_dir = os.path.join(MODEL_DIR, "dqn")
        os.makedirs(dqn_model_dir, exist_ok=True)

        dqn_checkpoint = CheckpointCallback(
            save_freq=10_000,
            save_path=dqn_model_dir,
            name_prefix="dqn_hjb_checkpoint",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

        dqn = DQN(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=0.01,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=64,
            tau=0.1,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.7,   # slower epsilon decay
            exploration_final_eps=0.1,  # more exploration at the end
            verbose=1,
            tensorboard_log=os.path.join(LOG_DIR, "dqn"),
            # seed=42,
        )

        print("\n===== Training DQN =====")
        dqn.learn(
            total_timesteps=total_timesteps,
            callback=dqn_checkpoint
        )
        dqn_path = os.path.join(dqn_model_dir, "dqn_hjb_final")
        dqn.save(dqn_path)
        print(f"[DQN] Saved final model to: {dqn_path}")

        self._simple_rollout(dqn, train_env, algo_name="DQN")

        # --------- PPO ---------
        ppo_model_dir = os.path.join(MODEL_DIR, "ppo")
        os.makedirs(ppo_model_dir, exist_ok=True)

        ppo_checkpoint = CheckpointCallback(
            save_freq=10_000,
            save_path=ppo_model_dir,
            name_prefix="ppo_hjb_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

        ppo = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            tensorboard_log=os.path.join(LOG_DIR, "ppo"),
            # seed=42,
        )

        print("\n===== Training PPO =====")
        ppo.learn(
            total_timesteps=total_timesteps,
            callback=ppo_checkpoint
        )
        ppo_path = os.path.join(ppo_model_dir, "ppo_hjb_final")
        ppo.save(ppo_path)
        print(f"[PPO] Saved final model to: {ppo_path}")

        self._simple_rollout(ppo, train_env, algo_name="PPO")

        # --------- A2C (additional baseline) ---------
        a2c_model_dir = os.path.join(MODEL_DIR, "a2c")
        os.makedirs(a2c_model_dir, exist_ok=True)

        a2c_checkpoint = CheckpointCallback(
            save_freq=10_000,
            save_path=a2c_model_dir,
            name_prefix="a2c_hjb_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

        a2c = A2C(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=7e-4,
            gamma=0.99,
            n_steps=5,
            verbose=1,
            tensorboard_log=os.path.join(LOG_DIR, "a2c"),
            # seed=42,
        )

        print("\n===== Training A2C =====")
        a2c.learn(
            total_timesteps=total_timesteps,
            callback=a2c_checkpoint
        )
        a2c_path = os.path.join(a2c_model_dir, "a2c_hjb_final")
        a2c.save(a2c_path)
        print(f"[A2C] Saved final model to: {a2c_path}")

        self._simple_rollout(a2c, train_env, algo_name="A2C")

        train_env.close()


'''if __name__ == "__main__":
    trainer = Train(DATA_PATH)
    # You can bump this up once things are stable
    trainer.train_all(total_timesteps=10_000)'''
