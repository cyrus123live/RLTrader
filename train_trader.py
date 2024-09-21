from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import StockData
import logging
from stable_baselines3.common.monitor import Monitor


MODEL_NAME = "trading_model"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_log.log"),
        logging.StreamHandler()
    ]
)

model = PPO.load(MODEL_NAME)

# Counter to control evaluation frequency
train_runs_before_eval = 10 
current_run = 0

while True:

    logging.info("Starting a new training iteration...")

    data = StockData.get_random_month()
    env = Monitor(TradingEnv(data))

    model.set_env(env)
    model.learn(total_timesteps=data.shape[0] - 1, progress_bar=True)
        
    model.save("trading_model")

    logging.info("Training complete, model saved successfully.")

    # current_run += 1
    # if current_run % train_runs_before_eval == 0:
    #     logging.info(f"Running evaluation...")
    #     test_model(model)