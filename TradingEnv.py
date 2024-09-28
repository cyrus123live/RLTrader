import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df
        self.reward_range = (-np.inf, np.inf)

        self.starting_cash = 10000000
        self.k = int(self.starting_cash / df['Close'].iloc[0]) # Maximum amount of stocks bought or sold each minute
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            #low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32 # Model 24 and below use this
            low=-1, high=1, shape=(self.df[self.df.filter(regex='_Scaled$').columns].shape[1] + 2,), dtype=np.float32
        ) 
        
        # Set starting point
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        # Initialize portfolio
        self.cash = self.starting_cash
        self.stock = 0 
        self.total_value = self.cash

        self.c_selling = 0
        self.c_buying = 0 # 0.5% fees is 0.0005
        self.r = 0
        self.r_bh = math.log(1-self.c_buying)
        self.last_action = 0

    def _get_obs(self):

        # current_row = self.df[["Close_Normalized", "Change_Normalized", "D_HL_Normalized"]].iloc[self.current_step] # Model 25 and below
        current_row = self.df[self.df.filter(regex='_Scaled$').columns].iloc[self.current_step]
        close = self.df["Close"].iloc[self.current_step]

        # amount_held = float((self.stock * close) / (self.stock * close + self.cash))
        # amount_held = self.stock / self.k 
        amount_held = np.clip(2 * self.stock / self.k - 1, -1, 1)
        
        # cash_normalized = self.cash / self.starting_cash
        cash_normalized = np.clip(2 * self.cash / self.starting_cash - 1, -1, 1)
        return np.array(current_row.tolist() + [amount_held, cash_normalized])

    def _take_action(self, action):

        current_price = self.df['Close'].iloc[self.current_step]

        # Current meta seems to be action space is -k, ..., 0,..., k for k stocks, normalized to [-1, 1], see finrl single stock example: https://finrl.readthedocs.io/en/latest/tutorial/Introduction/SingleStockTrading.html

        if action[0] > 0 and self.cash > 0: # buy

            # buyable_stocks = (self.cash) / (current_price * 1.004) 
            buyable_stocks = (self.cash) / (current_price * (1 + self.c_buying)) 

            to_buy = min(buyable_stocks, self.k * action[0])

            self.stock += to_buy
            self.cash -= to_buy * current_price * (1 + self.c_buying)

            self.last_buy_step = self.current_step
            self.last_action = 1

        elif action[0] < 0 and self.stock > 0: # sell

            # to_sell = min(self.stock, self.k * action[0] * -1)

            # self.stock -= to_sell
            # # self.cash += to_sell * (current_price * 0.996)
            # self.cash += to_sell * current_price
            # self.last_sell_step = self.current_step
            # self.last_action = -1
            self.cash += self.stock * current_price * (1 - self.c_selling)
            self.stock = 0

            self.last_sell_step = self.current_step
            self.last_action = -1 # This was commented out before?!

        else:
            self.last_action = 0

    def step(self, action):
        self._take_action(action)
        
        # Move to the next time step
        self.current_step += 1
        
        # Calculate portfolio value at the new step
        new_price = self.df['Close'].iloc[self.current_step]
        new_total_value = self.cash + self.stock * new_price

        # Implementing compounded excess return reward function
        r_f = 0.000041 / (60*6.5) # rough risk free rate is 1.5% per annum, or 0.000041 per day

        self.r_bh += math.log(self.df.iloc[self.current_step]["Close"]) - math.log(self.df.iloc[self.current_step - 1]["Close"])

        if self.last_action == 1:
            self.r += math.log(self.df.iloc[self.current_step]["Close"]) - math.log(self.df.iloc[self.current_step - 1]["Close"])
            self.r += math.log(1-self.c_buying)
        elif self.last_action == -1:
            self.r += r_f
            self.r += math.log(1-self.c_selling)

        reward = self.r - self.r_bh
        # reward = new_total_value - self.total_value * (2**-11) # reward scaling taken from https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/meta/env_stock_trading/env_stock_trading.py
        
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
        self.total_value = self.cash
        self.current_step = 0
        self.r = 0
        self.r_bh = math.log(1-self.c_buying)
        self.last_action = 0
        
        return self._get_obs(), {}

    def render(self, mode='human', close=False):
        if mode == 'human':
            # print(f"Step: {self.current_step}, Total Value: {self.total_value}, Cash: {self.cash}, Stocks: {self.stock}")
            return {"Portfolio_Value": self.total_value, "Close": self.df['Close'].iloc[self.current_step]}
