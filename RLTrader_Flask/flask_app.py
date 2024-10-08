from flask import Flask, request, render_template, session, redirect, url_for
import os
from dotenv import load_dotenv
import sqlite3 as sql
# from flaskr.auth import login_required
import datetime

app = Flask(__name__)
load_dotenv()
app.secret_key = os.getenv("APP_SECRET_KEY")

@app.route('/')
def index():
    if 'name' in session and session['name'] == "admin":

        conn = sql.connect("RLTrader.db")
        values = [[c[0] * c[2] + c[1], datetime.datetime.fromtimestamp(c[3]).day] for i, c in enumerate(conn.execute("SELECT close, cash, held, timestamp FROM trades").fetchall())]

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