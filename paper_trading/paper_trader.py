from paper_trading.wallet_manager import (
    load_wallet,
    update_balance,
    update_gold,
    update_silver,
    add_history,
)


# ==========================
# BUY
# ==========================

def buy_metal(metal, price, grams):

    wallet = load_wallet()

    cost = price * grams

    if wallet["balance"] < cost:
        print("Not enough balance")
        return

    update_balance(-cost)

    if metal == "Gold":
        update_gold(grams)

    if metal == "Silver":
        update_silver(grams)

    add_history({
        "type": "BUY",
        "metal": metal,
        "price": price,
        "grams": grams,
        "cost": cost
    })

    print(f"BUY {metal} {grams}g at {price}")


# ==========================
# SELL
# ==========================

def sell_metal(metal, price, grams):

    wallet = load_wallet()

    value = price * grams

    if metal == "Gold":

        if wallet["gold_grams"] < grams:
            print("Not enough gold")
            return

        update_gold(-grams)

    if metal == "Silver":

        if wallet["silver_grams"] < grams:
            print("Not enough silver")
            return

        update_silver(-grams)

    update_balance(value)

    add_history({
        "type": "SELL",
        "metal": metal,
        "price": price,
        "grams": grams,
        "value": value
    })

    print(f"SELL {metal} {grams}g at {price}")


# ==========================
# HOLD
# ==========================

def hold_trade(metal):

    add_history({
        "type": "HOLD",
        "metal": metal
    })

    print(f"HOLD {metal}")