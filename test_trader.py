from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import StockData
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
import time
import datetime
import sqlite3 as sql

counter = 0
with open("model_counter.txt", 'r') as f:
    counter = int(f.read())

MODEL_NAME = f"trading_model_{counter}"
MODEL_NAME = f"trading_model Backup 2"

def plot_history(history):
    plt.subplot(2, 1, 1)

    plt.title('Stock Movement')
    plt.plot([h["Close"] for h in history])

    plt.subplot(2, 1, 2)

    plt.title("Portfolio Value")
    plt.plot([h["Portfolio_Value"] for h in history])

    plt.show()

test_data = StockData.get_test_data()
model = PPO.load(MODEL_NAME)
k = 10000000 / test_data.iloc[0]["Close"]
held = 0
cash = 10000000
history = []

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

if input("Plot history? (Y/n): ") == "Y":
    plot_history(history)
else:
    print(f"Final Portfolio Value: {cash + held * test_data.iloc[-1]['Close']}")

quit()


test_data = StockData.get_test_data()
# test_data = StockData.get_year(8)

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

    plot_history(history)

test_model(PPO.load("trading_model"))