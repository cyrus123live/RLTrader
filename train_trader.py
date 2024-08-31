from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd


data = pd.read_csv('BTC_Hourly.csv')

processed_data = pd.DataFrame(index=data.index)

processed_data["Close"] = data["Close"]
processed_data["Change"] = data["Close"].diff()
processed_data["D_HL"] = data["High"] - data["Low"]

for feature in processed_data.columns:
    rolling_mean = processed_data[feature].rolling(window=20).mean()
    rolling_std = processed_data[feature].rolling(window=20).std()

    # Normalize the feature (subtract rolling mean, divide by rolling std dev)
    processed_data[f'{feature}_Normalized'] = (processed_data[feature] - rolling_mean) / rolling_std

processed_data.dropna(inplace=True)



quit()


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
