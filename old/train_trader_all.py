from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import StockData
import logging
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Figure
import json

MODEL_NAME = f"models/trading_model_15"
model = PPO.load(MODEL_NAME)

for year in range(0, 24):
    for month in range(1, 13):
        data = StockData.get_month(year, month)
        env = Monitor(TradingEnv(data))
        print(f"Year: {year}, Month: {month}")
        model.set_env(env)
        model.learn(total_timesteps=data.shape[0] - 1, progress_bar=True)

model.save(MODEL_NAME)