from stable_baselines3 import PPO
from TradingEnv import TradingEnv
import StockData
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import Monitor
import time
import datetime
import requests
import os
import sqlite3 as sql
from dotenv import load_dotenv

STARTING_CASH = 100000
EXAMPLE_CLOSE = 580
MODEL_NAME = "trading_model Backup 1"


load_dotenv()
api_key = os.getenv("API_KEY")
api_secret_key = os.getenv("API_SECRET_KEY")

def get_cash():
    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}
    response = requests.get("https://paper-api.alpaca.markets/v2/account", headers=headers)

    return float(response.json()["cash"])

def get_position():
    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}
    response = requests.get("https://paper-api.alpaca.markets/v2/positions", headers=headers)
    return response.json()

def get_position_quantity():
    if len(get_position()) > 0:
        return float(get_position()[0]['qty'])
    else:
        return 0

def get_position_value():
    if len(get_position()) > 0:
        return float(get_position()[0]['market_value'])
    else:
        return 0

def make_order(qty, buy_or_sell):

    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}
    params = {
        'symbol': 'SPY',
        'qty': str(qty),
        'side': buy_or_sell, # "buy" or "sell"
        'type': 'market',
        'time_in_force': 'day' # Note: experiment with this
    }

    response = requests.post("https://paper-api.alpaca.markets/v2/orders", headers=headers, json=params)
    return response.json()

def sell_all():
    return make_order(get_position_quantity(), "sell")

def buy(qty):
    return make_order(qty, "buy")


def main():

    conn = sql.connect("RLTrader.db")
    # conn.execute("DROP TABLE trades")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            close REAL,
            cash REAL,
            action REAL,
            held REAL
        )'''
    )

    model = PPO.load(MODEL_NAME)
    k = STARTING_CASH / EXAMPLE_CLOSE
    held = get_position_quantity()
    cash = get_cash()

    print(f"Starting trader session, cash: {cash}, held: {held}\n")
    
    while True:

        time.sleep(1)
        current_time = datetime.datetime.now()

        if current_time.hour > 13 or (current_time.hour == 13 and current_time.minute >= 1):
            print("Trading day over, ending trader session.")
            quit()

        if current_time.hour < 7:
            continue

        if current_time.second == 1: # every 1st second of each minute
            data = StockData.get_current_data()
            if data.shape[0] == 0:
                continue
            # print(data)

            # Obs: ["Close_Normalized", "Change_Normalized", "D_HL_Normalized", amount_held, cash_normalized]
            #   Amount held is normalized to k, starting cash / first close price
            #   Cash is normalized to starting cash 
            #   Resets each month during training, each year in testing
            obs = np.array(data[["Close_Normalized", "Change_Normalized", "D_HL_Normalized"]].iloc[-1].tolist() + [held / k, cash / STARTING_CASH])

            action = model.predict(obs, deterministic=True)[0][0]

            if action < 0:
                sell_all()
                print(f"{current_time.hour}:{current_time.minute:02d} Executing Sell")
            else:
                to_buy = min(cash / data.iloc[-1]["Close"], action * k)
                buy(to_buy)
                print(f"{current_time.hour}:{current_time.minute:02d} Executing Buy {to_buy}")

            time.sleep(15)

            # Update database
            cash = get_cash()
            held = get_position_quantity()

            conn.execute('''
                INSERT INTO trades (timestamp, close, cash, action, held) VALUES (?, ?, ?, ?, ?) ''', (
                datetime.datetime.now().timestamp(), # datetime.datetime.fromtimestamp() to reverse
                data.iloc[-1]["Close"], 
                cash, 
                float(action), 
                held
            ))
            conn.commit()
            print(f"Cash: {cash}, Held: {held}\n")

if __name__ == "__main__":
    main()