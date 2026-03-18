import json
from pathlib import Path

WALLET_FILE = Path("paper_trading/wallet.json")


def load_wallet():

    if not WALLET_FILE.exists():
        return None

    with open(WALLET_FILE, "r") as f:
        return json.load(f)


def save_wallet(data):

    with open(WALLET_FILE, "w") as f:
        json.dump(data, f, indent=4)


def add_history(entry):

    wallet = load_wallet()

    wallet["history"].append(entry)

    save_wallet(wallet)


def update_balance(amount):

    wallet = load_wallet()

    wallet["balance"] += amount

    save_wallet(wallet)


def update_gold(grams):

    wallet = load_wallet()

    wallet["gold_grams"] += grams

    save_wallet(wallet)


def update_silver(grams):

    wallet = load_wallet()

    wallet["silver_grams"] += grams

    save_wallet(wallet)