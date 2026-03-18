# ==========================================
# VALIDATION ENGINE
# ==========================================

from pathlib import Path
import json


USER_SETTINGS = Path("automation_engine/user_settings.json")
PORTFOLIO_FILE = Path("paper_trading/portfolio.json")


# ==========================================
# LOAD FILES
# ==========================================

def load_settings():

    with open(USER_SETTINGS, "r") as f:
        return json.load(f)


def load_portfolio():

    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)


# ==========================================
# GRAM VALIDATION
# Only multiples of 1 gram allowed
# ==========================================

def validate_grams(grams):

    if grams <= 0:
        return False, "Grams must be > 0"

    if grams % 1 != 0:
        return False, "Only multiples of 1 gram allowed"

    return True, "OK"


# ==========================================
# BUY RANGE VALIDATION
# must be within 2000 below current price
# ==========================================

def validate_buy_range(current_price, buy_price):

    if buy_price > current_price:
        return False, "Buy price must be below current price"

    if (current_price - buy_price) > 2000:
        return False, "Buy price must be within ₹2000 below market"

    return True, "OK"


# ==========================================
# SELL RANGE VALIDATION
# must be within 10000 above current price
# ==========================================

def validate_sell_range(current_price, sell_price):

    if sell_price < current_price:
        return False, "Sell price must be above current price"

    if (sell_price - current_price) > 10000:
        return False, "Sell price must be within ₹10000 above market"

    return True, "OK"


# ==========================================
# WALLET CHECK
# ==========================================

def validate_wallet(grams, price, metal):

    portfolio = load_portfolio()

    cost = grams * price

    if portfolio["balance"] < cost:
        return False, "Not enough balance"

    return True, "OK"


# ==========================================
# HOLD BUY CHECK
# if prediction shows downtrend
# ==========================================

def should_hold_buy(predicted_prices):

    if predicted_prices[-1] < predicted_prices[0]:
        return True

    return False


# ==========================================
# HOLD SELL CHECK
# if prediction shows uptrend
# ==========================================

def should_hold_sell(predicted_prices):

    if predicted_prices[-1] > predicted_prices[0]:
        return True

    return False