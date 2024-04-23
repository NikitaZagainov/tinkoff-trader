import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from gymnasium import Env
from gymnasium.spaces import Box


class MultiStockTradingEnv(Env):

    def __init__(self, df: pd.DataFrame, features: list[str], comission: float,
                 limits: list[int], reward_scaling: int,
                 initial_balance: float):
        self.df = df.dropna()
        self.features = features
        self.comission = comission
        self.limits = np.array(limits)
        self.reward_scaling = reward_scaling
        self.initial_balance = initial_balance

        self.df = self.df.sort_index()
        self.scaler = StandardScaler()
        self.df = self.df.unstack()
        self.scaler.fit(self.df)
        self.df = self.df.stack(future_stack=True)

        self.dates = self.df.index.get_level_values(0).unique()
        self.tics = self.df.index.get_level_values(1).unique()

        self.action_space = Box(low=-1,
                                high=1,
                                shape=[len(self.tics)],
                                dtype=np.float32)

        self.observation_space = Box(
            low=-float("inf"),
            high=float("inf"),
            shape=[len(self.tics) * (len(self.features) + 1) + 1],
            dtype=np.float32)

        self.epsilon = 0.0001

        self.reset()

    def reset(self, *args, **kwargs):
        self.cur_idx = 0
        self.shares = [0] * len(self.tics)
        self.portfolio_prev = self.initial_balance
        self.balance = self.initial_balance
        self.state = self.scaler.transform(np.array(
            self.df.unstack() \
                .iloc[self.cur_idx][self.features]).reshape(1, -1)).reshape(-1)
        self.state = np.concatenate(
            [np.zeros(shape=(len(self.tics), )), self.state], axis=0)
        self.state = np.concatenate(
            [np.array([self.portfolio_prev]), self.state], axis=0)

        self.state[0] /= self.initial_balance
        self.state[1:1 + len(self.tics)] /= self.limits

        return self.state, {}

    def step(self, action: np.array, *args, **kwargs):
        self.cur_idx += 1
        if (self.cur_idx >= len(self.dates)):
            return self.state, 0, True, True, {}

        action = (action * self.limits).astype(int)
        sell_ids = np.where(action < 0)[0]
        buy_ids = np.where(action > 0)[0]
        prices = self.df.unstack().iloc[self.cur_idx]["close"].values
        self.state[1 + len(self.tics):] = self.scaler.transform(
            self.df.unstack().iloc[self.cur_idx].values.reshape(
                1, -1)).reshape(-1)

        for idx in sell_ids:
            current = self.shares[idx]
            price = prices[idx]
            sell_amount = min(action[idx], current)
            self.shares[idx] -= sell_amount
            self.balance += sell_amount * price * (1 - self.comission -
                                                   self.epsilon)

        for idx in buy_ids:
            current = self.shares[idx]
            price = prices[idx]
            limit = self.limits[idx]
            buy_amount = int(
                min(
                    action[idx], limit - current, self.balance /
                    (price * (1 + self.comission * self.epsilon))))
            self.shares[idx] += buy_amount
            self.balance -= buy_amount * price * (1 + self.comission +
                                                  self.epsilon)

        self.state[0] = self.balance / self.initial_balance
        self.state[1:len(self.tics) + 1] = self.shares / self.limits

        portfolio_cur = self.calculate_portfolio(prices)

        reward = (portfolio_cur - self.portfolio_prev) * self.reward_scaling

        self.portfolio_prev = portfolio_cur

        return self.state, reward, False, False, {}

    def calculate_portfolio(self, prices):
        portfolio = self.balance
        for i in range(len(prices)):
            portfolio += prices[i] * self.shares[i]

        return portfolio
