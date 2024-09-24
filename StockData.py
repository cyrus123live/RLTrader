import yfinance as yf
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import random
import requests
import datetime as dt
from dotenv import load_dotenv
import os


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

    folder_name = "spy_data"

    data = pd.read_csv(f"{folder_name}/20{year:02d}-{month:02d}.csv", index_col="timestamp").iloc[::-1]
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

def get_day(year, month, day):
    month = get_month(year, month)
    return month[month.index.day == day]

def get_current_data():

    prices = yf.Ticker("SPY").history(period='max', interval='1m', prepost=True)
    prices["close"] = prices["Close"]
    prices["low"] = prices["Low"]
    prices["high"] = prices["High"]

    prices = prices[prices.index.day == int(dt.datetime.today().strftime("%d"))] # Get current date data

    return process_data(prices)
    

def get_current_alpaca():

    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_secret_key = os.getenv("API_SECRET_KEY")

    url = "https://data.alpaca.markets/v2/stocks/bars?symbols=spy&timeframe=1Min&limit=3000&adjustment=raw&feed=sip&sort=asc"
    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}

    response = requests.get(url, headers=headers) 

    data = pd.DataFrame.from_dict(response.json()['bars']['SPY'])

    data['time'] = pd.to_datetime(data['t'], format='%Y-%m-%dT%H:%M:%SZ')
    data['time'] = data['time'].dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    data = data.set_index('time')

    data['close'] = data['c']
    data['high'] = data['h']
    data['low'] = data['l']

    return process_data(data)