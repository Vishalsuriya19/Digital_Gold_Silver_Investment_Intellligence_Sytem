# ==========================================
# AUTO INVESTMENT ENGINE
# ==========================================

import sys
from pathlib import Path

# FIX PATH
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from automation_engine.decision_engine import run_decision
from automation_engine.signals import *


# ==========================================
# RUN ENGINE
# ==========================================

def run_auto_engine():

    print("\n==============================")
    print("AUTO INVESTMENT ENGINE STARTED")
    print("==============================")

    for metal in ["Gold", "Silver"]:

        print(f"\nChecking {metal}...")

        signal, message = run_decision(metal)

        print(f"Signal: {signal}")
        print(f"Message: {message}")

        if signal == BUY:
            print(f"{metal} BUY executed")

        elif signal == SELL:
            print(f"{metal} SELL executed")

        elif signal == HOLD_BUY:
            print(f"{metal} HOLD BUY")

        elif signal == HOLD_SELL:
            print(f"{metal} HOLD SELL")

        elif signal == HOLD:
            print(f"{metal} HOLD")

        elif signal == ERROR:
            print(f"{metal} ERROR")

        else:
            print(f"{metal} NO TRADE")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    run_auto_engine()