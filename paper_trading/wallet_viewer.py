import json
from pathlib import Path

WALLET_FILE = Path("paper_trading/wallet.json")


def load_wallet():

    if not WALLET_FILE.exists():
        print("Wallet not found")
        return None

    with open(WALLET_FILE, "r") as f:
        return json.load(f)


def show_wallet():

    wallet = load_wallet()

    if wallet is None:
        return

    print("\n===== WALLET =====")

    print("Balance:", wallet["balance"])
    print("Gold grams:", wallet["gold_grams"])
    print("Silver grams:", wallet["silver_grams"])

    print("\nHistory:")

    for h in wallet["history"]:
        print(h)