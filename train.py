import os
import numpy as np
import pandas as pd
import pandas_ta as ta

from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

from drl_agent.portfolio_alloc_env import PortfolioAllocationEnv


def main():
    data = pd.read_csv("./data/data.csv", index_col=0, header=[0, 1])
    data = data.dropna().stack(future_stack=True).sort_index(level=0)

    data["rsi_6"] = data.groupby(level=1)["close"].transform(
        lambda x: ta.rsi(x, length=6)
    )
    data["rsi_12"] = data.groupby(level=1)["close"].transform(
        lambda x: ta.rsi(x, length=12)
    )
    data["rsi_24"] = data.groupby(level=1)["close"].transform(
        lambda x: ta.rsi(x, length=24)
    )

    data["macd"] = data.groupby(level=1)["close"].transform(
        lambda x: ta.macd(x).iloc[:, 0]
    )

    data["boll_ub"] = data.groupby(level=1)["close"].transform(
        lambda x: ta.bbands(np.log1p(x), length=20).iloc[:, 2]
    )
    data["boll_lb"] = data.groupby(level=1)["close"].transform(
        lambda x: ta.bbands(np.log1p(x), length=20).iloc[:, 0]
    )
    data["boll_m"] = data.groupby(level=1)["close"].transform(
        lambda x: ta.bbands(np.log1p(x), length=20).iloc[:, 1]
    )

    data["garman_klass_vol"] = (
        (np.log(data["high"]) - np.log(data["low"])) ** 2
    ) / 2 - (2 * np.log(2) - 1) * ((np.log(data["close"]) - np.log(data["open"])) ** 2)

    def atr(data):
        atr = ta.atr(data["high"], data["low"], data["close"], length=14)
        return atr

    data["atr"] = data.groupby(level=1, group_keys=False).apply(atr)

    def kdj(data):
        kdj = ta.kdj(data["high"], data["low"], data["close"])
        return kdj

    data["kdjk"] = data.groupby(level=1, group_keys=False).apply(kdj).iloc[:, 0]
    data["kdjd"] = data.groupby(level=1, group_keys=False).apply(kdj).iloc[:, 1]
    data["kdjj"] = data.groupby(level=1, group_keys=False).apply(kdj).iloc[:, 2]

    features = list(data.columns.get_level_values(0).unique())
    features.remove("volume")

    train_data = data[data.index.get_level_values(0) < "2024-01-01"]
    train_data = train_data[train_data.index.get_level_values(0) > "2023-01-01"]
    test_data = data[data.index.get_level_values(0) >= "2024-01-01"]

    train_env = PortfolioAllocationEnv(
        df=train_data,
        features=features,
        commission=5e-4,
        reward_scaling=1e-5,
        initial_balance=1_000_000,
    )

    ppo_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./models/ppo",
        name_prefix="ppo_periodical_short_mem",
        verbose=1,
    )
    ppo = RecurrentPPO(
        RecurrentActorCriticPolicy,
        train_env,
        verbose=1,
        policy_kwargs={"n_lstm_layers": 3},
        n_steps=512,
        clip_range=0.3,
    )

    model = ppo.learn(80000, callback=ppo_callback)
    model.save("./models/ppo/last_short_mem.zip")


if __name__ == "__main__":
    main()
