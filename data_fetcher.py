# ==========================================
# LIVE METAL PRICE FETCHER (INDIA FIXED)
# ==========================================

import requests
from datetime import datetime
from config import API_KEY, API_URL

OUNCE_TO_GRAM = 31.1035


def fetch_today_price():
    """
    Fetch Gold & Silver price in INR per gram
    """

    params = {
        "api_key": API_KEY,
        "base": "USD",
        "currencies": "XAU,XAG,INR"
    }

    try:
        response = requests.get(API_URL, params=params, timeout=10)

        if response.status_code != 200:
            raise Exception(response.text)

        data = response.json()

        rates = data["rates"]

        gold_rate = rates["XAU"]
        silver_rate = rates["XAG"]
        usd_inr = rates["INR"]

        # convert to USD per ounce if needed
        if gold_rate < 1:
            gold_rate = 1 / gold_rate

        if silver_rate < 1:
            silver_rate = 1 / silver_rate

        # USD/ounce → INR/gram
        gold_inr = (gold_rate * usd_inr) / OUNCE_TO_GRAM
        silver_inr = (silver_rate * usd_inr) / OUNCE_TO_GRAM

        return {
            "date": datetime.today().strftime("%Y-%m-%d"),
            "gold": round(gold_inr, 2),
            "silver": round(silver_inr, 2)
        }

    except Exception as e:
        print("Price fetch error:", e)
        return None


if __name__ == "__main__":
    print(fetch_today_price())