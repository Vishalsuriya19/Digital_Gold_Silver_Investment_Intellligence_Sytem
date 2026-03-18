# broker_api/paper_broker.py

from paper_trading.paper_trader import buy_metal, sell_metal


class PaperBroker:

    def buy(self, metal, price, grams):

        ok, msg = buy_metal(metal, price, grams)

        return ok, msg

    def sell(self, metal, price, grams):

        ok, msg = sell_metal(metal, price, grams)

        return ok, msg

    def balance(self):

        return "paper"