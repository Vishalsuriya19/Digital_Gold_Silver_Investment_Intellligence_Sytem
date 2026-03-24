# broker_api/broker_factory.py

from broker_api.paper_broker import PaperBroker
from broker_api.zerodha_broker import ZerodhaBroker
from broker_api.upstox_broker import UpstoxBroker


BROKER = "paper"


def get_broker():

    if BROKER == "paper":
        return PaperBroker()

    if BROKER == "zerodha":
        return ZerodhaBroker()

    if BROKER == "upstox":
        return UpstoxBroker()

    return PaperBroker()