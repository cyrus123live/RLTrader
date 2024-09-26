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
from pathlib import Path

CASH_DIVISOR = 10
STARTING_CASH = 100000 / CASH_DIVISOR
EXAMPLE_CLOSE = 580
MODEL_NAME = "models/trading_model Backup 1"


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

def cancel_all():
    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}
    response = requests.delete("https://paper-api.alpaca.markets/v2/orders", headers=headers)
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

def make_order(qty, buy_or_sell, price):

    headers = {"accept": "application/json", "APCA-API-KEY-ID": api_key, "APCA-API-SECRET-KEY": api_secret_key}
    params = {
        'symbol': 'SPY',
        'qty': str(qty),
        'side': buy_or_sell, # "buy" or "sell"
        'type': 'limit',
        'limit_price': str(price),
        'time_in_force': 'day' # Note: experiment with this
    }

    response = requests.post("https://paper-api.alpaca.markets/v2/orders", headers=headers, json=params)
    return response.json()

def sell_all(price):
    return make_order(get_position_quantity(), "sell", price)

def buy(qty, price):
    return make_order(qty, "buy", price)


def main():

    conn = sql.connect("/root/RLTrader/RLTrader.db")
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

    model = PPO.load("/root/RLTrader/" + MODEL_NAME)
    k = STARTING_CASH / EXAMPLE_CLOSE
    held = get_position_quantity()
    cash = get_cash()  / CASH_DIVISOR

    print(f"Starting trader session, cash: {cash}, held: {held}\n")
    
    while True:

        try:

            time.sleep(1)
            current_time = datetime.datetime.now()

            if current_time.weekday() == 5 or current_time.weekday() == 6:
                print("It is the weekend, ending trader session.")
                quit()

            if current_time.hour > 20 or (current_time.hour == 20 and current_time.minute >= 1):
                print("Trading day over, ending trader session.")
                quit()

            if current_time.hour < 11:
                continue

            if current_time.second == 1: # every 1st second of each minute
                try:
                    data = StockData.get_current_data()
                except Exception as e:
                    print("Error in getting current data:", e)
                    continue

                if data.shape[0] == 0:
                    continue

                # Obs: ["Close_Normalized", "Change_Normalized", "D_HL_Normalized", amount_held, cash_normalized]
                #   Amount held is normalized to k, starting cash / first close price
                #   Cash is normalized to starting cash 
                #   Resets each month during training, each year in testing
                obs = np.array(data[["Close_Normalized", "Change_Normalized", "D_HL_Normalized"]].iloc[-1].tolist() + [held / k, cash / STARTING_CASH])

                action = model.predict(obs, deterministic=True)[0][0]

                if action < 0:
                    print(sell_all(data['Close'].iloc[-1]), "\n\n")
                    print(f"{current_time.hour}:{current_time.minute:02d} Executed sell at price {data['Close'].iloc[-1]}")
                else:
                    to_buy = min(cash / data.iloc[-1]["Close"], action * k)
                    if to_buy > 0:
                        print(buy(to_buy, data['Close'].iloc[-1]), "\n\n")
                        print(f"{current_time.hour}:{current_time.minute:02d} Executed buy {to_buy} at price {data['Close'].iloc[-1]}")
                    else:
                        print("Tried to buy a negative amount")

                time.sleep(25)
                cancel_all() # cancel orders if not made in 25 seconds, so that we can get up to date info and safely move to next minute
                time.sleep(5)

                # Update database
                cash = get_cash()  / CASH_DIVISOR
                held = get_position_quantity()

                value = get_position_value()
                print(f"New value of stock portfolio: {value}")

                conn.execute('''
                    INSERT INTO trades (timestamp, close, cash, action, held) VALUES (?, ?, ?, ?, ?) ''', (
                    datetime.datetime.now().timestamp(), # datetime.datetime.fromtimestamp() to reverse
                    data.iloc[-1]["Close"], 
                    cash * CASH_DIVISOR, 
                    float(action), 
                    held
                ))
                conn.commit()
                print(f"Cash: {cash}, Held: {held}\n")

        except Exception as e:
            print("Failure in loop:", e)

if __name__ == "__main__":
    main()