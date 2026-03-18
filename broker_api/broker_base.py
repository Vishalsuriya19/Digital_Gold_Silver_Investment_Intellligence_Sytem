# broker_api/broker_base.py

class BrokerBase:

    def buy(self, metal, price, grams):
        raise NotImplementedError

    def sell(self, metal, price, grams):
        raise NotImplementedError

    def balance(self):
        raise NotImplementedError