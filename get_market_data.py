import os
import sys
import asyncio
import logging
from datetime import timedelta
from trader import Trader
from strategies.drl_strategy import DRLStrategy
from tinkoff.invest import AsyncClient, CandleInterval
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX
from stable_baselines3 import A2C

TOKEN = os.environ["INVEST_TOKEN"]
ACCOUNT_ID = os.environ["ACCOUNT_ID"]

TICKERS = [
    "YNDX", "TCSG", "PLZL", "NVTK", "MGNT", "LKOH", "OZON", "SMLT", "AGRO",
    "TRNFP", "BELU", "CHMF", "SFIN", "LSRG", "BANEP", "HHRU", "VSMO", "GCHE",
    "BANE", "AKRN"
]
FEATURES = [
    'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_30_sma',
    'close_60_sma', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'donchian_min', 'donchian_max', 'donchian_mean', 'VWAP_M', 'ISA_9',
    'ISB_26', 'ITS_9', 'IKS_26'
]


async def main():
    async with AsyncClient(TOKEN, target=INVEST_GRPC_API_SANDBOX, app_name="") as client:
        model = A2C.load("./models/a2c_low_no_five/best_model.zip")
        strategy = DRLStrategy(TICKERS, timedelta(days=20),
                               CandleInterval.CANDLE_INTERVAL_DAY, model)
        trader = Trader(client, ACCOUNT_ID, strategy)
        await trader.init_trader(1_200_000)
        
        data = await trader.get_historical_data()
        trader.logger.info(data)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        filename="datastream.log",
        filemode="w",
        format="%(levelname)s %(name)s %(asctime)s %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S")
    logging.getLogger("tinkoff.invest.logging").setLevel(logging.WARNING)
    with open("/dev/null", "w") as f:
        sys.stdout = f
        asyncio.run(main())
