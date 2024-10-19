from flask import Flask, jsonify, request
import requests
import pandas as pd

app = Flask(__name__)

# Alpha Vantage API key
API_KEY = 'your_alpha_vantage_api_key'

def get_stock_data(symbol):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=1min&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    if "Error Message" in data:
        return {"error": "Stock symbol not found or API limit reached"}

    time_series = data.get('Time Series (1min)', {})
    df = pd.DataFrame(time_series).transpose()
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    df.index = pd.to_datetime(df.index)
    return df.head(5).to_dict()

@app.route('/stock/<symbol>', methods=['GET'])
def stock(symbol):
    data = get_stock_data(symbol)
    if "error" in data:
        return jsonify(data), 400
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
