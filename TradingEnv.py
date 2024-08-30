import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()
    
        self.df = df
        self.reward_range = (-np.inf, np.inf)
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        ) 
        
        # Set starting point
        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        # Initialize portfolio
        self.cash = 1000000
        self.stock = 0 
        self.total_value = self.cash + self.stock * self.df['Close_Not_Normalized'].iloc[0]
        self.portfolio_values = [self.total_value]

        self.last_sell_step = 0
        self.last_buy_step = 0

    def _get_obs(self):
        current_row = self.df.iloc[self.current_step]
        amount_held = float((self.stock * current_row["Close_Not_Normalized"]) / (self.stock * current_row["Close_Not_Normalized"] + self.cash))
        
        return np.array(current_row.tolist() + [amount_held])

    def _take_action(self, action):

        current_price = self.df['Close_Not_Normalized'].iloc[self.current_step]

        if action[0] > 0 and self.cash > 0: # buy
            buyable_stocks = (self.cash) // (current_price * 1.004)
            self.stock += buyable_stocks * action[0]
            self.cash -= buyable_stocks * action[0] * (current_price * 1.004)
            self.last_buy_step = self.current_step

        elif action[0] < 0 and self.stock > 0: # sell

            to_sell = self.stock * action[0] * -1
            self.stock -= to_sell
            self.cash += to_sell * (current_price * 0.996)
            self.last_sell_step = self.current_step

    def step(self, action):
        self._take_action(action)
        
        # Move to the next time step
        self.current_step += 1
        
        # Calculate portfolio value at the new step
        new_price = self.df['Close_Not_Normalized'].iloc[self.current_step]
        new_total_value = self.cash + self.stock * new_price
        
        # Update the total value and portfolio history
        self.total_value = new_total_value
        self.portfolio_values.append(self.total_value)

        # Normalized Portfolio Growth
        normalized_portfolio_value = self.total_value / self.portfolio_values[0]  # Normalize to starting value
        growth_reward = normalized_portfolio_value - 1  # Measure growth rate

        # Sortino Ratio Reward
        if len(self.portfolio_values) > 1:
            returns = np.diff(self.portfolio_values)  # Change in portfolio values
            mean_return = np.mean(returns)
            downside_returns = [r for r in returns if r < 0]
            downside_deviation = np.std(downside_returns) if downside_returns else 0
            risk_adjusted_reward = (mean_return / downside_deviation) if downside_deviation else 0
        else:
            risk_adjusted_reward = 0

        # Combine Growth and Risk-Adjusted Reward
        reward = 0.7 * growth_reward + 0.3 * risk_adjusted_reward  # Weighted sum
        reward += 0.01 if np.abs(action)[0] > 0 else 0

        # Define whether the episode is finished
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get next observation
        obs = self._get_obs()
        info = {'total_value': self.total_value}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.cash = 1000000
        self.stock = 0
        self.current_step = 0
        self.total_value = self.cash + self.stock * self.df['Close_Not_Normalized'].iloc[0]
        self.portfolio_values = [self.total_value]
        
        return self._get_obs(), {}

    def render(self, mode='human', close=False):
        if mode == 'human':
            # print(f"Step: {self.current_step}, Total Value: {self.total_value}, Cash: {self.cash}, Stocks: {self.stock}")
            return {"Portfolio_Value": self.total_value, "Close": self.df['Close_Not_Normalized'].iloc[self.current_step]}
