from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from TradingEnv import TradingEnv
from StockData import StockData
import numpy as np
import matplotlib.pyplot as plt

sd = StockData("BTC-USD")
df = sd.get_training_data().iloc[::-1]

train_size = int(len(df) * 0.8)  # 80% of the data for training

# print(df.head(20))

# Split the data
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

test_env = TradingEnv(test_df)

model = SAC.load("trading_model")

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