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


counter = 0
info = ""
with open("model_counter.txt", 'r') as f:
    counter = int(f.read())

with open('model_info.txt', 'r') as f:
    info = json.loads(f.read())

MODEL_NAME = f"trading_model_{counter}"
TIMESTEPS = 50000

model = PPO.load(MODEL_NAME)
best_result = 0

if str(counter) in info:
    best_result = info[str(counter)]['best_result']
best_result_flag = False

global steps
steps = model.num_timesteps

def test_model(test_data):
    history = []
    k = 10000000 / test_data.iloc[0]["Close"]
    cash = 10000000
    held = 0
    test_env = Monitor(TradingEnv(test_data))
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

    return history

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.steps = 0
        self.plot_interval = 30000
        self.plot_large_interval = 50000

    def _on_step(self):
        global steps 
        steps += 1

        # Large test (8 months)
        if steps % self.plot_large_interval == 0:
            print("Performing test...")

            test_data = StockData.get_test_data()
            history = test_model(test_data)

            to_plot = pd.DataFrame(index=test_data.index)
            to_plot['close'] = [h["Close"]/history[0]["Close"] for h in history]
            to_plot['portfolio'] = [h["Portfolio_Value"]/history[0]['Portfolio_Value'] for h in history]

            figure = plt.figure()
            p = figure.add_subplot()

            p.plot(to_plot['close'], label="Stock Movement")
            p.plot(to_plot['portfolio'], label="Portfolio Value")
            p.legend()

            self.logger.record("images/large_test", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()

            # Set best result flag for saving model
            global best_result
            global best_result_flag
            if history[-1]['Portfolio_Value'] > best_result:
                best_result = history[-1]['Portfolio_Value']
                best_result_flag = True

        return True


while True:

    logging.info("Starting a new training iteration...")

    data = StockData.get_random_month()
    env = Monitor(TradingEnv(data))

    model.set_env(env)
    model.learn(total_timesteps=data.shape[0] - 1, progress_bar=True, 
    callback=TensorboardCallback(), 
    reset_num_timesteps=False)
        
    if best_result_flag:
        best_result_flag = False
        model.save(MODEL_NAME)

        if str(counter) not in info:
            info[str(counter)] = {}
        info[str(counter)]['best_result'] = best_result
        info[str(counter)]['steps'] = steps

        print("model saved")
        with open('model_info.txt', 'w') as f:
            json.dump(info, f)

    logging.info("Training complete, model saved successfully.")
