import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.starting_cash = 10000000
        self.k = int(self.starting_cash / df['Close'].iloc[0]) # Maximum amount of stocks bought or sold each minute
    
        self.df = df
        self.reward_range = (-np.inf, np.inf)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        ) 
        
        # Set starting point
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        # Initialize portfolio
        self.cash = self.starting_cash
        self.stock = 0 
        self.total_value = self.cash

        self.last_sell_step = 0
        self.last_buy_step = 0

    def _get_obs(self):
        current_row = self.df[["Close_Normalized", "Change_Normalized", "D_HL_Normalized"]].iloc[self.current_step]
        close = self.df.iloc[self.current_step]["Close"]

        amount_held = float((self.stock * close) / (self.stock * close + self.cash))
        cash_normalized = self.cash / self.starting_cash
        
        return np.array(current_row.tolist() + [amount_held, cash_normalized])

    def _take_action(self, action):

        current_price = self.df['Close'].iloc[self.current_step]

        # Current meta seems to be action space is -k, ..., 0,..., k for k stocks, normalized to [-1, 1], see finrl single stock example: https://finrl.readthedocs.io/en/latest/tutorial/Introduction/SingleStockTrading.html

        if action[0] > 0 and self.cash > 0: # buy

            buyable_stocks = (self.cash) / (current_price * 1.004) # Note that this implies float stock amounts

            to_buy = min(buyable_stocks, self.k * action[0])

            self.stock += to_buy
            self.cash -= to_buy * (current_price * 1.004)
            self.last_buy_step = self.current_step

        elif action[0] < 0 and self.stock > 0: # sell

            to_sell = min(self.stock, self.k * action[0] * -1)

            self.stock -= to_sell
            self.cash += to_sell * (current_price * 0.996)
            self.last_sell_step = self.current_step

    def step(self, action):
        self._take_action(action)
        
        # Move to the next time step
        self.current_step += 1
        
        # Calculate portfolio value at the new step
        new_price = self.df['Close'].iloc[self.current_step]
        new_total_value = self.cash + self.stock * new_price

        reward = new_total_value - self.total_value * (2**-11) # reward scaling taken from https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/meta/env_stock_trading/env_stock_trading.py
        
        # Update the total value and portfolio history
        self.total_value = new_total_value

        # Define whether the episode is finished
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get next observation
        obs = self._get_obs()
        info = {'total_value': self.total_value}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.cash = self.starting_cash
        self.stock = 0
        self.current_step = 0
        self.total_value = self.cash
        
        return self._get_obs(), {}

    def render(self, mode='human', close=False):
        if mode == 'human':
            # print(f"Step: {self.current_step}, Total Value: {self.total_value}, Cash: {self.cash}, Stocks: {self.stock}")
            return {"Portfolio_Value": self.total_value, "Close": self.df['Close'].iloc[self.current_step]}
