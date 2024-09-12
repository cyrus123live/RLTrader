from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import StockData
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor

def plot_history(history):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax.set_title('Stock Movement')
    ax.plot([h["Close"] for h in history], label='Closing Price')

    ax2.set_title("Portfolio Value")
    ax2.plot([h["Portfolio_Value"] for h in history], label='Portfolio Value')

    plt.show()

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