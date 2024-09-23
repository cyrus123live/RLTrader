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


counter = 0
with open("model_counter.txt", 'r') as f:
    counter = int(f.read())

MODEL_NAME = f"trading_model_{counter}"
TIMESTEPS = 50000

model = PPO.load(MODEL_NAME)

global steps
steps = model.num_timesteps

test_data = StockData.get_month(24, 8)
test_env = Monitor(TradingEnv(test_data))
def test_model():
    history = []
    k = 10000000 / test_data.iloc[0]["Close"]
    cash = 10000000
    held = 0
    for i in range(test_data.shape[0]):

        data = test_data.iloc[i]

        # Obs: ["Close_Normalized", "Change_Normalized", "D_HL_Normalized", amount_held, cash_normalized]
        #   Amount held is normalized to k, starting cash / first close price
        #   Cash is normalized to starting cash 
        #   Resets each month during training, each year in testing
        obs = np.array(data[["Close_Normalized", "Change_Normalized", "D_HL_Normalized"]].tolist() + [held / k, cash / 10000000])

        action = model.predict(obs, deterministic=True)[0][0]

        if action < 0:
            cash += held * data["Close"]
            held = 0
        else:
            to_buy = min(cash / data["Close"], action * k)
            cash -= to_buy * data["Close"]
            held += to_buy

        history.append({"Portfolio_Value": cash + held * data["Close"], "Close": data["Close"], "Cash": cash, "Held": held})

    return [h["Portfolio_Value"] for h in history], [h["Close"] for h in history]

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.steps = 0
        self.plot_interval = 30000

    def _on_step(self):
        global steps 
        steps += 1

        if steps == 1:
            figure = plt.figure()
            figure.add_subplot().plot(test_model()[1])
            # Close the figure after logging it
            self.logger.record("trajectory/stock_price", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()
        
        if steps % self.plot_interval == 0:
            figure = plt.figure()
            figure.add_subplot().plot(test_model()[0])
            # Close the figure after logging it
            self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()
        return True


while True:

    logging.info("Starting a new training iteration...")

    data = StockData.get_random_month()
    env = Monitor(TradingEnv(data))

    model.set_env(env)
    model.learn(total_timesteps=data.shape[0] - 1, progress_bar=True, 
    callback=TensorboardCallback(), 
    reset_num_timesteps=False)
        
    model.save(MODEL_NAME)

    logging.info("Training complete, model saved successfully.")
