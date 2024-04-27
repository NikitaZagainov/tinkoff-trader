import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from gymnasium import Env
from gymnasium.spaces import Box, Space


class PortfolioAllocationEnv(Env):

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

        self.action_space = Box(low=-10,
                                high=10,
                                shape=[len(self.tics)],
                                dtype=np.float32)

        self.observation_space = Box(
            low=-float("inf"),
            high=float("inf"),
            shape=[len(self.tics) * len(self.features)],
            dtype=np.float32)

        self.epsilon = 0.0001

        self.reset()

    def step(self, action, *args, **kwargs):
        self.cur_idx += 1
        if self.cur_idx == len(self.dates):
            return self.state, 0, True, True, {}
        
        action = softmax(action)
        
        prices = self.df.unstack().iloc[self.cur_idx]["close"].values
        
        self.portfolio_buffer.append(self.calculate_portfolio(prices))
        available_money = self.portfolio_buffer[-1]
        available_shares = np.floor(available_money * action) 
        
        to_buy = np.floor(available_shares / prices / (1 + self.comission + self.epsilon))
        delta = to_buy - self.shares
        
        sell_ids = np.where(delta < 0)[0]
        buy_ids = np.where(delta > 0)[0]
        
        for idx in sell_ids:
            self.balance += (-1) * prices[idx] * delta[idx] * (1 - self.comission - self.epsilon)
            self.shares[idx] += delta[idx]
            
        for idx in buy_ids:
            self.balance -= prices[idx] * delta[idx] * (1 + self.comission + self.epsilon)
            self.shares[idx] += delta[idx]
            
        self.state = self.scaler.transform(
            self.df.unstack().iloc[self.cur_idx].values.reshape(
                1, -1)).reshape(-1)
        
        reward = (self.portfolio_buffer[-1] - self.portfolio_buffer[-2]) * self.reward_scaling
        return self.state, reward, False, False, {}

    def reset(self, *args, **kwargs):
        self.cur_idx = 0
        self.portfolio_buffer = [self.initial_balance]
        self.balance = self.initial_balance
        self.shares = np.array([0] * len(self.tics), dtype=np.float32)
        self.state = self.scaler.transform(np.array(
            self.df.unstack() \
                .iloc[self.cur_idx][self.features]).reshape(1, -1)).reshape(-1)
        
        return self.state, {}
    
    def calculate_portfolio(self, prices):
        portfolio = self.balance
        for i in range(len(self.shares)):
            portfolio += prices[i] * self.shares[i] * (1 - self.comission - self.epsilon)
        return portfolio
