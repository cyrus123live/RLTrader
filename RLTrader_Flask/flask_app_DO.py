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

@app.route('/minutely')
def index():
    if 'name' in session and session['name'] == "admin":

        # conn = sql.connect("/root/RLTrader/RLTrader.db")
        folder_name = datetime.datetime.now().strftime("%Y-%m-%d")
        df = pd.read_csv(f"/root/RLTrader/csv/{folder_name}/minutely.csv")
        values = [[df.iloc[i]["Close"] * df.iloc[i]["Held"] + df.iloc[i]["Cash"], datetime.datetime.fromtimestamp(c[3]).day] for i in range(len(df))]
        # values = [[c[0] * c[2] + c[1], datetime.datetime.fromtimestamp(c[3]).day] for i, c in enumerate(conn.execute("SELECT close, cash, held, timestamp FROM trades").fetchall())]

        return render_template("index.html", name=session['name'], data=[v[0] for v in values], labels=[v[1] for v in values])
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