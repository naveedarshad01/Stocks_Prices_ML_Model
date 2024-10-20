from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import time
import numpy as np
import joblib
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Yahoo Finance stock symbol and parameters
symbol = 'AAPL'  # Replace with your desired stock symbol
interval = '1min'

# Initialize the machine learning model and scaler
scaler = StandardScaler()

# Path to save/load the model
MODEL_FILE = 'stock_model_yfinance.pkl'

# Function to load or initialize model
def load_or_initialize_model():
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        print("Model loaded from file.")
    else:
        model = SGDRegressor(max_iter=1000, tol=1e-3)
        print("New model initialized.")
    return model

# Load or initialize the model
model = load_or_initialize_model()

# Function to fetch live stock data from Yahoo Finance
def get_live_stock_data(symbol):
    stock_data = yf.Ticker(symbol)
    # Fetch the last 1 day data at 1 minute interval
    df = stock_data.history(interval="1m", period="1d")

    if df.empty:
        print("Failed to fetch data or unknown error.")
        return None

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df

# Preprocessing function to generate features
def preprocess_data(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    df = df.dropna()  # Remove rows with missing values due to rolling calculations
    return df

# Online training function
def train_model_online(X, y):
    X_scaled = scaler.fit_transform(X)  # Normalize input features
    model.partial_fit(X_scaled, y)  # Online learning (incremental training)

# Function to save the trained model
def save_model():
    joblib.dump(model, MODEL_FILE)
    print("Model saved successfully.")

# Fetch, preprocess, and train model continuously in the background
def continuously_fetch_and_train():
    while True:
        df = get_live_stock_data(symbol)
        if df is not None:
            df = preprocess_data(df)
            X = df[['MA5', 'MA10', 'HL_PCT', 'PCT_change', 'Volume']].values
            y = df['Close'].values

            train_model_online(X, y)
            print("Model trained on live data batch.")

            # Save the model after each training iteration
            save_model()

        time.sleep(60)  # Wait 60 seconds before fetching more data

# Flask route to predict based on live data
@app.route('/predict/<symbol>', methods=['GET'])
def predict(symbol):
    df = get_live_stock_data(symbol)
    if df is not None:
        df = preprocess_data(df)
        X = df[['MA5', 'MA10', 'HL_PCT', 'PCT_change', 'Volume']].values
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)
        return jsonify({"symbol": symbol, "predicted_price": prediction[-1]})
    return jsonify({"error": "Failed to fetch stock data"}), 500

if __name__ == '__main__':
    # Start live training in the background (non-blocking)
    import threading
    t = threading.Thread(target=continuously_fetch_and_train)
    t.start()

    # Start Flask app
    app.run(debug=True)
