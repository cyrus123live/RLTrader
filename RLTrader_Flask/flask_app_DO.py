from flask import Flask, request, render_template, session, redirect, url_for
import os
from dotenv import load_dotenv
import sqlite3 as sql
# from flaskr.auth import login_required
import datetime
import pandas as pd

app = Flask(__name__)
load_dotenv()
app.secret_key = os.getenv("APP_SECRET_KEY")


@app.route('/minutely_json')
def minutely_jason():

    try:
        folder_name = datetime.datetime.now().strftime("%Y-%m-%d")
        df = pd.read_csv(f"/root/RLTrader/csv/{folder_name}/minutely.csv")
    except:
        folder_name = datetime.datetime(year = datetime.datetime.now().year, month = datetime.datetime.now().month, day = datetime.datetime.now().day - 1).strftime("%Y-%m-%d")
        df = pd.read_csv(f"/root/RLTrader/csv/{folder_name}/minutely.csv")

    return df.to_json()

@app.route('/minutely')
def index():
    if 'name' in session and session['name'] == "admin":

        # conn = sql.connect("/root/RLTrader/RLTrader.db")
        try:
            folder_name = datetime.datetime.now().strftime("%Y-%m-%d")
            df = pd.read_csv(f"/root/RLTrader/csv/{folder_name}/minutely.csv")
        except:
            folder_name = datetime.datetime(year = datetime.datetime.now().year, month = datetime.datetime.now().month, day = datetime.datetime.now().day - 1).strftime("%Y-%m-%d")
            df = pd.read_csv(f"/root/RLTrader/csv/{folder_name}/minutely.csv")
        
        values = [[float((df.iloc[i]["Close"] * df.iloc[i]["Resulting Held"] + df.iloc[i]["Resulting Cash"]) / (df.iloc[0]["Close"] * df.iloc[0]["Resulting Held"] + df.iloc[0]["Resulting Cash"])), float(datetime.datetime.fromtimestamp(df.iloc[i]["Time"]).strftime("%H%M"))] for i in range(len(df))]
        closes = [[float((df.iloc[i]["Close"]) / (df.iloc[0]["Close"])), float(datetime.datetime.fromtimestamp(df.iloc[i]["Time"]).strftime("%H%M"))] for i in range(len(df))]
        missed_buys = [float(datetime.datetime.fromtimestamp(df.iloc[i]["Time"]).strftime("%H%M")) for i in range(len(df[df["Missed Buy"] == True]))]
        missed_sells = [float(datetime.datetime.fromtimestamp(df.iloc[i]["Time"]).strftime("%H%M")) for i in range(len(df[df["Missed Sell"] == True]))]
        # values = [[c[0] * c[2] + c[1], datetime.datetime.fromtimestamp(c[3]).day] for i, c in enumerate(conn.execute("SELECT close, cash, held, timestamp FROM trades").fetchall())]

        return render_template("index.html", name=session['name'], data=[v[0] for v in values], closes=[v[0] for v in closes], missed_buys = missed_buys, missed_sells = missed_sells, labels=[v[1] for v in values])
    return render_template("index.html")

@app.route("/daily")
def daily_graph():
    if 'name' in session and session['name'] == "admin":

        conn = sql.connect("/root/RLTrader/RLTrader.db")
        values = [[(c[0] - c[1]) / c[0], datetime.datetime.fromtimestamp(c[2]).day] for i, c in enumerate(conn.execute("SELECT total_trades, missed_trades, timestamp FROM days").fetchall())]

        return render_template("index.html", name=session['name'], data=[v[0] for v in values], labels=[v[1] for v in values])
    return render_template("index.html")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
         
        if request.form['Password'] == "password":
            session['name'] = 'admin'
        else:
            return render_template("login.html")

        return redirect(url_for('index'))
    return render_template("login.html")

@app.route('/logout')
def logout():
    session.pop('name', None)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)