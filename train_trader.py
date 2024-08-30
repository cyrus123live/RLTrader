from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from TradingEnv import TradingEnv
from StockData import StockData
import numpy as np
import matplotlib.pyplot as plt


TRAINING_ROUNDS = 8

sd = StockData("BTC-USD")
df = sd.get_training_data().iloc[::-1]

train_size = int(len(df) * 0.8)  # 80% of the data for training

model = SAC.load("trading_model")

# Split the data
train_df = df.iloc[:train_size]

for i in range(TRAINING_ROUNDS):

    index = len(train_df) / TRAINING_ROUNDS

    env = TradingEnv(train_df[int(index * i) : int(index * (i + 1))])

    model.set_env(env)  # Set the model to the current environment
    model.learn(total_timesteps=len(train_df) * 4)  # Train for a specified number of timesteps # TODO: replace train_df
    model.save("trading_model")
