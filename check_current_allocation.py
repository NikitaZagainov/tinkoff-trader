import os
import asyncio

from tinkoff.invest import AsyncClient, OrderType, OrderDirection
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX


TOKEN = os.environ["INVEST_TOKEN"]
ACCOUNT_ID = os.environ["ACCOUNT_ID"]


async def main():
    async with AsyncClient(TOKEN, target=INVEST_GRPC_API_SANDBOX) as client:
        shares = await client.operations.get_portfolio(account_id=ACCOUNT_ID)
        for item in shares.positions:
            if "RUB" in item.figi:
                continue
            print(item)


if __name__ == "__main__":
    asyncio.run(main())
