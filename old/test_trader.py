from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import StockData
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
import time
import datetime
import sqlite3 as sql
import pandas as pd

counter = 0
with open("models/model_counter.txt", 'r') as f:
    counter = int(f.read())

# FEES_PER_SHARE = 0.0035
# MINIMUM_FEE = 0.25
FEES_PER_SHARE = 0
MINIMUM_FEE = 0

STARTING_CASH = 1000000
MODEL_NAME = f"models/trading_model_{counter}"
MODEL_NAME = f"models/trading_model_24"
test_data = StockData.get_current_data()
# test_data = StockData.get_year(8)

model = PPO.load(MODEL_NAME)
held = 0
cash = STARTING_CASH
k = STARTING_CASH / test_data.iloc[0]["Close"]
history = []

for i in range(test_data.shape[0]):

    data = test_data.iloc[i]

    # Obs: ["Close_Normalized", "Change_Normalized", "D_HL_Normalized", amount_held, cash_normalized]
    #   Amount held is normalized to k, starting cash / first close price
    #   Cash is normalized to starting cash 
    #   Resets each month during training, each year in testing
    obs = np.array(data[["Close_Normalized", "Change_Normalized", "D_HL_Normalized"]].tolist() + [held / k, cash / STARTING_CASH]) # 24 and below use this
    # obs = np.array(test_data[test_data.filter(regex='_Scaled$').columns].iloc[i].tolist() + [np.clip(2 * held / k - 1, -1, 1), np.clip(2 * cash / STARTING_CASH - 1, -1, 1)])

    action = model.predict(obs, deterministic=True)[0][0]

    if action < 0 and cash + held * data["Close"] >= max(MINIMUM_FEE, FEES_PER_SHARE * held):
        cash += held * data["Close"] - max(MINIMUM_FEE, FEES_PER_SHARE * held)
        held = 0
    else:
        to_buy = action * k
        while to_buy * data["Close"] + max(MINIMUM_FEE, FEES_PER_SHARE * to_buy) > cash:
            to_buy -= 1
        if to_buy < 0:
            to_buy = 0

        if to_buy != 0:
            cash -= to_buy * data["Close"] + max(MINIMUM_FEE, FEES_PER_SHARE * to_buy)
            held += to_buy

    # 0.1% fees
    # if action < 0:
    #     cash += held * data["Close"] * 0.999
    #     held = 0
    # else:
    #     to_buy = min(cash / data["Close"] * 1.001, action * k)
    #     cash -= to_buy * data["Close"] * 1.001
    #     held += to_buy

    history.append({"Portfolio_Value": cash + held * data["Close"], "Close": data["Close"], "Cash": cash, "Held": held})

f = plt.figure()
f.set_figheight(8)
f.set_figwidth(10)

plot = pd.DataFrame(index=test_data.index)
plot['close'] = [h["Close"]/history[0]['Close'] for h in history]
plot['portfolio'] = [h["Portfolio_Value"]/history[0]['Portfolio_Value'] for h in history]

plt.plot(plot['close'], label="Close")
# plt.plot(test_data['D_HL_Normalized'], label="D_HL_Normalized")
# plt.plot(test_data['Change_Normalized'], label="Change_Normalized")
# plt.plot(test_data['Close_Normalized'], label="Close_Normalized")
plt.plot(plot['portfolio'], label="Portfolio Value", color='tab:red')

plt.legend()
plt.show()

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