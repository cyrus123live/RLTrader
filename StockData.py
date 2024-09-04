import yfinance as yf
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import random


def process_data(data):

    processed_data = pd.DataFrame(index=data.index)

    processed_data["Close"] = data["close"]
    processed_data["Change"] = data["close"].diff()
    processed_data["D_HL"] = data["high"] - data["low"]

    for feature in processed_data.columns:
        rolling_mean = processed_data[feature].rolling(window=20).mean()
        rolling_std = processed_data[feature].rolling(window=20).std()

        # Normalize the feature (subtract rolling mean, divide by rolling std dev)
        processed_data[f'{feature}_Normalized'] = (processed_data[feature] - rolling_mean) / rolling_std

    processed_data.dropna(inplace=True)

    return processed_data.between_time('07:00', '16:00')

def get_month(year, month):
    data = pd.read_csv(f"spy_data/20{year:02d}-{month:02d}.csv", index_col="timestamp").iloc[::-1]
    data.index = pd.to_datetime(data.index)
    return process_data(data)

def get_random_month():
    return get_month(random.randint(0, 23), random.randint(1, 12))

def get_test_data():
    frames = []
    for i in range(1, 9):
        frames.append(get_month(24, i))
    return pd.concat(frames)

def get_year(year):
    frames = []
    for i in range(1, 13):
        frames.append(get_month(year, i))
    return pd.concat(frames)
