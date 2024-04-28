from strategies.base_strategy import BaseStrategy
from datetime import datetime, timedelta
import logging
import pandas as pd
import asyncio
from decimal import Decimal

from tinkoff.invest import (AsyncClient, HistoricCandle, CandleInterval,
                            CandleInstrument, OrderType, OrderDirection,
                            PostOrderResponse, MoneyValue,
                            SandboxPayInResponse, PortfolioResponse)
from tinkoff.invest.utils import quotation_to_decimal, decimal_to_quotation
from tinkoff.invest.services import InstrumentsService
from tinkoff.invest.async_services import AsyncServices, AsyncMarketDataStreamManager
from tinkoff.invest.exceptions import AioRequestError


class Trader:

    def __init__(
        self,
        client: AsyncServices,
        account_id: str,
        strategy: BaseStrategy,
    ):
        self.client = client
        self.account_id = account_id
        self.strategy = strategy
        self.ticker_list = strategy.get_ticker_list()
        self.historical_period, self.trading_interval = (
            strategy.get_historical_candles_config())
        self.logger = logging.getLogger(name="robot")

    async def init_trader(self, starting_money: int):
        self.logger.info("Initializing trader")
        self.ticker_list, self.figi_list, self.lot_list = await self.get_figi(
            self.ticker_list)
        self.figi2ticker = {
            f: t
            for f, t in zip(self.figi_list, self.ticker_list)
        }
        self.ticker2figi = {
            t: f
            for t, f in zip(self.ticker_list, self.figi_list)
        }

        portfolio = await self.get_portfolio()
        current_portfolio = quotation_to_decimal(
            portfolio.total_amount_portfolio)

        if current_portfolio > 0:
            self.logger.warning(
                "Current portfolio already has %s on balance. " +
                "Method will not add money to account", current_portfolio)
        else:
            await self.add_money(starting_money)
            self.logger.info("Portfolio initialized")

    async def get_portfolio(self) -> PortfolioResponse:
        portfolio = await self.client.operations.get_portfolio(
            account_id=self.account_id)
        return portfolio

    async def add_money(self,
                        amount: int,
                        currency: str = "rub") -> SandboxPayInResponse:
        amount = decimal_to_quotation(Decimal(amount))
        response = await self.client.sandbox.sandbox_pay_in(
            account_id=self.account_id,
            amount=MoneyValue(units=amount.units,
                              nano=amount.nano,
                              currency=currency),
        )

        return response

    async def get_figi(
            self, ticker_list: list) -> tuple[list[str], list[str], list[int]]:
        tickers = []
        instruments: InstrumentsService = self.client.instruments
        for method in ["shares", "bonds", "etfs", "currencies", "futures"]:
            result = await getattr(instruments, method)()
            for item in result.instruments:
                tickers.append({
                    "ticker": item.ticker,
                    "figi": item.figi,
                    "lot": item.lot
                })

        tickers_df = pd.DataFrame(tickers)
        ticker_df = tickers_df[tickers_df["ticker"].isin(ticker_list)]

        if ticker_df.empty:
            self.logger.warning(
                "Returning None: no correct ticker was passed to function")
            return None

        ticker_df = ticker_df.sort_values(["ticker"])

        tickers = ticker_df["ticker"].tolist()
        figi = ticker_df["figi"].tolist()
        lot = ticker_df["lot"].tolist()

        if len(tickers) < len(ticker_list):
            self.logger.warning(
                "Failed to get figi for some tickers: %s",
                ", ".join([tic for tic in ticker_list if tic not in tickers]))

        return tickers, figi, lot

    async def get_candles_history(self, figi: str) -> list[HistoricCandle]:
        candles = []
        async for candle in self.client.get_all_candles(
                figi=figi,
                from_=datetime.now() - self.historical_period,
                to=datetime.now(),
                interval=self.trading_interval):
            candles.append(candle)

        return candles

    def candles_to_df(self, candles: list, ticker: str = None) -> pd.DataFrame:

        def to_decimal(quotation):
            fractional = quotation["nano"] / Decimal("10e8")
            return Decimal(quotation["units"]) + fractional

        df = pd.DataFrame(candles)
        try:
            df["open"] = df["open"].apply(to_decimal).astype(float)
            df["high"] = df["high"].apply(to_decimal).astype(float)
            df["low"] = df["low"].apply(to_decimal).astype(float)
            df["close"] = df["close"].apply(to_decimal).astype(float)

            if ticker:
                df["tic"] = ticker
        except Exception:
            pass

        return df

    async def get_historical_data(self) -> pd.DataFrame:
        df = None
        for figi, tic in zip(self.figi_list, self.ticker_list):
            while True:
                try:
                    candles = await self.get_candles_history(figi)
                    break
                except AioRequestError as e:
                    await asyncio.sleep(e.metadata.ratelimit_reset)
            if df is None:
                df = self.candles_to_df(candles, tic)
            else:
                df = pd.concat([df, self.candles_to_df(candles, tic)], axis=0)

        return df.drop(columns=["is_complete", "candle_source"])

    async def buy(self, figi: str, quantity: int) -> PostOrderResponse:
        response = await self.client.market_data.get_trading_status(figi=figi)
        if not response.api_trade_available_flag or not response.market_order_available_flag:
            self.logger.error("Unable to buy %s: market closed", self.figi2ticker[figi])
            return response
        response = await self.client.orders.post_order(
            figi=figi,
            account_id=self.account_id,
            quantity=quantity,
            order_type=OrderType.ORDER_TYPE_BESTPRICE,
            direction=OrderDirection.ORDER_DIRECTION_BUY)
        return response

    async def sell(self, figi: str, quantity: int) -> PostOrderResponse:
        response = await self.client.market_data.get_trading_status(figi=figi)
        if not response.api_trade_available_flag or not response.market_order_available_flag:
            self.logger.error("Unable to sell %s: market closed", self.figi2ticker[figi])
            return response
        response = await self.client.orders.post_order(
            figi=figi,
            account_id=self.account_id,
            quantity=quantity,
            order_type=OrderType.ORDER_TYPE_BESTPRICE,
            direction=OrderDirection.ORDER_DIRECTION_SELL)
        return response

    async def get_current_balance(self):
        portfolio = await self.client.operations.get_portfolio(
            account_id=self.account_id)
        current_balance = float(
            quotation_to_decimal(portfolio.total_amount_currencies))
        return current_balance

    async def get_current_shares(self):
        portfolio = await self.client.operations.get_portfolio(
            account_id=self.account_id)
        positions = portfolio.positions
        shares = {self.figi2ticker[f]: 0 for f in self.figi_list}
        for position in positions:
            if position.figi not in self.figi_list:
                continue
            shares[self.figi2ticker[position.figi]] = float(
                quotation_to_decimal(position.quantity))
        return shares

    async def trading_step(self):
        self.logger.info("Performing trading step")
        df = await self.get_historical_data()

        balance = await self.get_current_balance()
        shares = await self.get_current_shares()

        actions = self.strategy.get_trading_action(df, balance, shares)
        self.logger.info("Actions performed by strategy: \n%s", actions)
        for ticker, action in actions.items():
            if ticker not in self.ticker2figi:
                continue

            figi = self.ticker2figi[ticker]
            if action > 0:
                self.logger.info("Bying %s of %s (%s)", action, ticker, figi)
                await self.buy(figi, action)
            elif action < 0:
                self.logger.info("Selling %s of %s (%s)", -action, ticker,
                                 figi)
                await self.sell(figi, -action)

    async def start_trading(self):
        market_data_stream: AsyncMarketDataStreamManager = \
            self.client.create_market_data_stream()

        subscriptions = [
            CandleInstrument(figi, interval=self.trading_interval)
            for figi in self.figi_list
        ]
        market_data_stream.candles.waiting_close().subscribe(subscriptions)

        last_timestamp = datetime.now()
        async for _ in market_data_stream:
            if datetime.now() - last_timestamp >= timedelta(seconds=10):
                await self.trading_step()
