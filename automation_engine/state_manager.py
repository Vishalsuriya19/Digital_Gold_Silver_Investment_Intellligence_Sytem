# ==========================================
# ENGINE STATE MANAGER
# ==========================================

import json
from pathlib import Path


PORTFOLIO_FILE = Path("paper_trading/portfolio.json")


# ===== STATES =====

IDLE = "IDLE"
WAITING_BUY = "WAITING_BUY"
WAITING_SELL = "WAITING_SELL"
BOUGHT = "BOUGHT"
SOLD = "SOLD"
COOLDOWN = "COOLDOWN"
HOLDING = "HOLDING"
ERROR = "ERROR"


# ==========================================
# LOAD PORTFOLIO
# ==========================================

def load_portfolio():

    if not PORTFOLIO_FILE.exists():
        raise FileNotFoundError("portfolio.json not found")

    with open(PORTFOLIO_FILE, "r") as f:
        return json.load(f)


# ==========================================
# SAVE PORTFOLIO
# ==========================================

def save_portfolio(data):

    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ==========================================
# GET STATE
# ==========================================

def get_state():

    data = load_portfolio()
    return data.get("engine_state", WAITING_BUY)


# ==========================================
# SET STATE
# ==========================================

def set_state(new_state):

    data = load_portfolio()
    data["engine_state"] = new_state
    save_portfolio(data)