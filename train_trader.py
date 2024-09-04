from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import StockData
import logging
from stable_baselines3.common.monitor import Monitor

def plot_history(history):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax.set_title('Stock Movement')
    ax.plot([h["Close"] for h in history], label='Closing Price')

    ax2.set_title("Portfolio Value")
    ax2.plot([h["Portfolio_Value"] for h in history], label='Portfolio Value')

    plt.show()


test_data = StockData.get_test_data()
test_env = Monitor(TradingEnv(test_data))
def test_model(model):
    history = []
    obs, info = test_env.reset()
    test_env.render()
    for _ in range(test_data.shape[0] - 1):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminate, truncated, info = test_env.step(action)
        history.append(test_env.render())
        if terminate or truncated:
            obs, info = test_env.reset()

    logging.info(f"Finished evaluation, final render: {history[-1]}")
    plot_history(history)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_log.log"),
        logging.StreamHandler()
    ]
)

model = PPO.load("trading_model")

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