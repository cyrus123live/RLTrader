from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import StockData

env = TradingEnv(StockData.get_random_month())

model = PPO("MlpPolicy", env, verbose=1)
model.save("trading_model")
