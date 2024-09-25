from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import StockData

counter = 0
with open("models/model_counter.txt", 'r') as f:
    counter = int(f.read())
with open("models/model_counter.txt", 'w') as f:
    f.write(str(counter + 1))

env = TradingEnv(StockData.get_random_month())

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=f"./tensorboard_logs/{counter + 1}/")
model.save(f"models/trading_model_{counter + 1}")
