# price_api/price_fetcher.py

import yfinance as yf


def get_gold_price():

    # Gold futures
    ticker = yf.Ticker("GC=F")

    data = ticker.history(period="1d")

    price = data["Close"].iloc[-1]

    return float(price)


def get_silver_price():

    ticker = yf.Ticker("SI=F")

    data = ticker.history(period="1d")

    price = data["Close"].iloc[-1]

    return float(price)


def get_price(metal):

    if metal == "Gold":
        return get_gold_price()

    if metal == "Silver":
        return get_silver_price()

    return None