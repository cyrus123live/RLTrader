from stable_baselines3 import PPO
from TradingEnv import TradingEnv
from StockData import StockData

env = TradingEnv({})

model = PPO("MlpPolicy", env, verbose=1)
model.save("trading_model")
