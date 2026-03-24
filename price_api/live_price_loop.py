import time
from price_api.price_fetcher import get_price
from automation_engine.decision_engine import run_decision


INTERVAL = 10  # seconds


def run_live():

    print("\nLIVE PRICE LOOP STARTED\n")

    while True:

        for metal in ["Gold", "Silver"]:

            price = get_price(metal)

            print(f"{metal} price:", price)

            signal, msg = run_decision(metal)

            print("Signal:", signal, msg)

        print("Waiting...", INTERVAL)

        time.sleep(INTERVAL)


if __name__ == "__main__":

    run_live()