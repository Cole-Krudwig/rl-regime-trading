import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


class Fetch:
    def __init__(self):
        self.av_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    def get_daily(self, symbol):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={self.av_api_key}'
        r = requests.get(url)
        data = r.json()
        return data

    def json_to_df(self, data):
        ts = data["Time Series (Daily)"]
        df = pd.DataFrame.from_dict(ts, orient="index")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })

        df = df.astype(float)

        return df

    def fetch(self, symbol):
        data = self.get_daily(symbol=symbol)
        df = self.json_to_df(data)
        df['Log_Return'] = np.log(
            df['close'] / df['close'].shift(1))
        df.to_csv(f'../data/{symbol}.csv')
        return df
