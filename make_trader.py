from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MlpPolicy
from TradingEnv import TradingEnv
from StockData import StockData


sd = StockData("BTC-USD")
df = sd.get_training_data()

train_size = int(len(df) * 0.8)  # 80% of the data for training

# Split the data
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

env = TradingEnv(train_df)

model = SAC(
    policy=MlpPolicy,
    env=env,
    learning_rate=0.001,
    learning_starts=2000,
    batch_size=512,
    tau=0.001, 
    gamma=0.99,  
    train_freq=1,  
    gradient_steps=1, 
    ent_coef=0.1,  # Entropy coefficient can be automatically adjusted
    target_entropy='auto',
    verbose=1
)
model.save("trading_model")
