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

CASH_DIVISOR = 100
STARTING_CASH = 100000 / CASH_DIVISOR
EXAMPLE_CLOSE = 580
MODEL_NAME = "models/trading_model_24"


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

def buy_all(price, cash):
    # to_buy = cash / data.iloc[-1]["Close"]
    to_buy = cash / price
    return make_order(to_buy, 'buy', price)

def buy(qty, price):
    return make_order(qty, "buy", price)

def end_trading_day(cash, held, starting_cash, starting_held, total_trades, missed_trades, data = StockData.get_current_data()):
    conn = sql.connect("/root/RLTrader/RLTrader.db")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS days (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            open REAL,
            close REAL,
            ending_held REAL,
            starting_held REAL,
            ending_cash REAL,
            starting_cash REAL,
            total_trades REAL,
            missed_trades REAL
        )'''
    )
    conn.execute("INSERT INTO days (timestamp, open, close, ending_held, starting_held, ending_cash, starting_cash, total_trades, missed_trades) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
        datetime.datetime.now().timestamp(),
        data["Close"].iloc[0],
        data["Close"].iloc[-1],
        held,
        starting_held,
        cash,
        starting_cash,
        total_trades,
        missed_trades
    ))
    conn.commit()


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

    # Parameters
    model = PPO.load("/root/RLTrader/" + MODEL_NAME)
    k = STARTING_CASH / EXAMPLE_CLOSE
    held = get_position_quantity()
    cash = get_cash()  / CASH_DIVISOR

    missed_trades = 0
    total_trades = 0
    starting_cash = cash
    starting_held = held

    print(f"Starting trader session, cash: {cash}, held: {held}\n")
    
    while True:

        try:

            time.sleep(1)
            current_time = datetime.datetime.now()

            # Weekend
            if current_time.weekday() == 5 or current_time.weekday() == 6:
                print("It is the weekend, ending trader session.")
                quit()

            # Too late
            if current_time.hour > 20 or (current_time.hour == 20 and current_time.minute >= 1):
                print("Trading day over, ending trader session.")
                end_trading_day(cash, held, starting_cash, starting_held, total_trades, missed_trades) # TODO: make sure this works
                print("Trading day ended successfully.")
                quit()

            # Too early
            if current_time.hour < 11:
                continue

            # every 1st second of each minute
            if current_time.second == 1: 
                try:
                    data = StockData.get_current_data()
                except Exception as e:
                    print("Error in getting current data:", e)
                    continue

                if data.shape[0] == 0:
                    continue

                obs = np.array(data[["Close_Normalized", "Change_Normalized", "D_HL_Normalized"]].iloc[-1].tolist() + [held / k, cash / STARTING_CASH])

                action = model.predict(obs, deterministic=True)[0][0]

                if action < 0 and held > 0:
                    total_trades += 1
                    print(sell_all(round(data['Close'].iloc[-1], 2)), "\n\n")
                    print(f"{current_time.strftime('%Y-%m-%d %H-%M')} Executed sell at price {round(data['Close'].iloc[-1], 2)}")
                elif action > 0 and cash > 10:
                    total_trades += 1
                    print(buy_all(round(data['Close'].iloc[-1], 2), cash), "\n\n")
                    print(f"{current_time.strftime('%Y-%m-%d %H-%M')} Executed buy all at price {round(data['Close'].iloc[-1], 2)}")
                else:
                    print(f"{current_time.strftime('%Y-%m-%d %H-%M')} Holding at price {round(data['Close'].iloc[-1], 2)}")

                time.sleep(25)
                if len(cancel_all()) > 0: # cancel orders if not made in 25 seconds, so that we can get up to date info and safely move to next minute
                    missed_trades += 1 # Not yet sure if this will work 
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