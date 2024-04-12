from abc import ABC
import pandas as pd
from datetime import datetime, timedelta
from tinkoff.invest import CandleInterval


class BaseStrategy(ABC):

    def get_ticker_list(self) -> list[str]:
        pass

    def get_historical_candles_config(
            self) -> tuple[timedelta, CandleInterval]:
        pass

    def get_trading_action(self, data: pd.DataFrame, available_money: float,
                           current_shares: dict[str, int]) -> dict[str, int]:
        pass
