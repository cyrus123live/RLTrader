from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import StockData
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
import time
import datetime
import sqlite3 as sql


def plot_history():

    closes1 = StockData.get_current_alpaca()['Close']
    closes2 = StockData.get_current_data()['Close']

    print(closes1)
    print(closes2)

    plt.subplot(1, 1, 1)

    plt.title('Stock Movement')
    plt.plot(closes1, alpha=0.5)
    plt.plot(closes2, alpha=0.5)

    plt.show()

plot_history()