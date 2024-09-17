from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import StockData
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
import time
import datetime
import sqlite3 as sql

def plot_history(history):
    plt.subplot(1, 2, 1)

    plt.title('Stock Movement')
    plt.plot([h["Close"] for h in history], label='Closing Price')

    plt.subplot(1, 2, 2)

    plt.title("Portfolio Value")
    plt.plot([h["Portfolio_Value"] for h in history], label='Portfolio Value')

    plt.show()

test_data = StockData.get_test_data()
model = PPO.load("trading_model Backup 1")
k = 10000000 / test_data.iloc[0]["Close"]
held = 0
cash = 10000000
history = []

conn = sql.connect("RLTrader_Test.db")
conn.execute("DROP TABLE trades")
conn.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        close REAL,
        cash REAL,
        action REAL,
        held REAL
    )'''
)

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
    conn.execute('''
        INSERT INTO trades (timestamp, close, cash, action, held) VALUES (?, ?, ?, ?, ?) ''', (
        data.name.timestamp(), # datetime.datetime.fromtimestamp() to reverse
        data["Close"], 
        cash, 
        action, 
        held
    ))
    conn.commit()

plot_history(history)


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

'''

sd = StockData("BTC-USD")
df = sd.get_training_data().iloc[::-1]

train_size = int(len(df) * 0.8)  # 80% of the data for training

# print(df.head(20))

# Split the data
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

test_env = TradingEnv(test_df)

model = PPO.load("trading_model")

history = []

obs, info = test_env.reset()
test_env.render()
for _ in range(len(test_df) - 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminate, truncated, info = test_env.step(action)
    history.append(test_env.render())
    if terminate or truncated:
        obs, info = test_env.reset()


fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax.set_title('Stock Movement')
ax.plot([h["Close"] for h in history], label='Closing Price')

ax2.set_title("Portfolio Value")
ax2.plot([h["Portfolio_Value"] for h in history], label='Portfolio Value')

plt.show()


# def plot_history(history):
#     fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#     ax.set_title('Stock Movement')
#     ax.plot([h["Close"] for h in history], label='Closing Price')

#     ax2.set_title("Portfolio Value")
#     ax2.plot([h["Portfolio_Value"] for h in history], label='Portfolio Value')

#     plt.show()


# test_data = StockData.get_test_data()
# test_env = TradingEnv(test_data)
# def test_model(model):
#     history = []
#     obs, info = test_env.reset()
#     test_env.render()
#     for _ in range(test_data.shape[0]):
#         action, _states = model.predict(obs, deterministic=True)
#         obs, reward, terminate, truncated, info = test_env.step(action)
#         history.append(test_env.render())
#         if terminate or truncated:
#             obs, info = test_env.reset()


'''