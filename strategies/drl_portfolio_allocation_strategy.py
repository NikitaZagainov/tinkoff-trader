from .base_strategy import BaseStrategy

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pandas_ta as ta
from scipy.special import softmax
import pickle

from sklearn.preprocessing import StandardScaler
import logging
from tinkoff.invest import CandleInterval
from stable_baselines3.common.base_class import BaseAlgorithm

FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "rsi_6",
    "rsi_12",
    "rsi_24",
    "macd",
    "boll_ub",
    "boll_lb",
    "boll_m",
    "garman_klass_vol",
    "atr",
    "kdjk",
    "kdjd",
    "kdjj",
]


class DRLPortfolioAllocationStrategy(BaseStrategy):
    def __init__(
        self,
        ticker_list: list[str],
        historical_interval: timedelta,
        trading_interval: CandleInterval,
        model: BaseAlgorithm,
    ):
        self.ticker_list = sorted(ticker_list)
        self.historical_interval = historical_interval
        self.trading_interval = trading_interval
        self.model = model
        self.logger = logging.getLogger("drl_portfolio_allocation_strategy")
        self.features = FEATURES

    def get_historical_candles_config(self) -> tuple[timedelta, CandleInterval]:
        return self.historical_interval, self.trading_interval

    def get_ticker_list(self) -> list[str]:
        return self.ticker_list

    def get_trading_action(
        self, 
        data: pd.DataFrame, 
        available_money: float, 
        current_shares: dict[str, int],
        prices: dict[str, float]
    ) -> dict[str, int]:
        data = data.pivot(columns="tic", index="time")
        data = self.add_indicators(data.dropna())
        
        state = self.compose_model_state(data)
        self.logger.info("current state: %s", state)
        lstm_state_old = self.get_lstm_state()
        action, lstm_state_new = self.model.predict(state, state=lstm_state_old, deterministic=True)
        self.save_lstm_state(lstm_state_new)
        self.logger.info("action: %s", action)
        
        portfolio = available_money
        for tic in self.ticker_list:
            portfolio += prices[tic] * current_shares[tic] * (1 - 5e-4)
        
        to_allocate = portfolio * (1 / (1 + np.exp(-action[0])))
        self.logger.info("investing %s out of %s", to_allocate, portfolio)
        
        allocation = softmax(action[1:]) * to_allocate
        prices_array = np.array([prices[tic] for tic in self.ticker_list])
        shares = np.floor(allocation / prices_array).astype(np.int32)
        self.logger.info("new shares allocation: %s", shares)
        
        delta = dict()
        for tic, share in zip(self.ticker_list, shares):
            if share - current_shares[tic] != 0:
                delta[tic] = share - current_shares
                
        self.logger.info("action: %s", delta)
        raise Exception("Not so fast! You are not tested yet!")
        return delta
        
        
    def compose_model_state(self, df: pd.DataFrame):
        scaler = StandardScaler()
        scaler.fit(df[FEATURES])
        today = pd.to_datetime(datetime.now().date())
        df.index = pd.to_datetime(df.index).tz_localize(None)
        
        state = scaler.transform(df.loc[today][FEATURES].values.reshape(1, -1)).reshape(-1)
        
        return state
        
    def get_lstm_state(self):
        try:
            with open("lstm_state.pkl", mode="rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            self.logger.warning("error reading from file: %s", e)
            return None
        
    def save_lstm_state(self, lstm_state):
        with open("lstm_state.pkl", mode="wb") as f:
            pickle.dump(lstm_state, f)

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.stack(future_stack=True)
        df["rsi_6"] = df.groupby(level=1)["close"].transform(
            lambda x: ta.rsi(x, length=6)
        )
        df["rsi_12"] = df.groupby(level=1)["close"].transform(
            lambda x: ta.rsi(x, length=12)
        )
        df["rsi_24"] = df.groupby(level=1)["close"].transform(
            lambda x: ta.rsi(x, length=24)
        )

        df["macd"] = df.groupby(level=1)["close"].transform(
            lambda x: ta.macd(x).iloc[:, 0]
        )

        df["boll_ub"] = df.groupby(level=1)["close"].transform(
            lambda x: ta.bbands(np.log1p(x), length=20).iloc[:, 2]
        )
        df["boll_lb"] = df.groupby(level=1)["close"].transform(
            lambda x: ta.bbands(np.log1p(x), length=20).iloc[:, 0]
        )
        df["boll_m"] = df.groupby(level=1)["close"].transform(
            lambda x: ta.bbands(np.log1p(x), length=20).iloc[:, 1]
        )

        df["garman_klass_vol"] = ((np.log(df["high"]) - np.log(df["low"])) ** 2) / 2 - (
            2 * np.log(2) - 1
        ) * ((np.log(df["close"]) - np.log(df["open"])) ** 2)

        def atr(data):
            atr = ta.atr(data["high"], data["low"], data["close"], length=14)
            return atr

        df["atr"] = df.groupby(level=1, group_keys=False).apply(atr)

        def kdj(data):
            kdj = ta.kdj(data["high"], data["low"], data["close"])
            return kdj

        df["kdjk"] = df.groupby(level=1, group_keys=False).apply(kdj).iloc[:, 0]
        df["kdjd"] = df.groupby(level=1, group_keys=False).apply(kdj).iloc[:, 1]
        df["kdjj"] = df.groupby(level=1, group_keys=False).apply(kdj).iloc[:, 2]
        
        return df.unstack().dropna()
