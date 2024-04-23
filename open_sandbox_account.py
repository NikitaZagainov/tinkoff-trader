import os
import asyncio

from tinkoff.invest import AsyncClient
from tinkoff.invest.constants import INVEST_GRPC_API_SANDBOX


TOKEN = os.environ["INVEST_TOKEN"]


async def main():
    async with AsyncClient(TOKEN, target=INVEST_GRPC_API_SANDBOX) as client:
        # accounts = await client.users.get_accounts()
        # for account in accounts.accounts:
        #     await client.sandbox.close_sandbox_account(account_id=account.id)
        
        account = await client.sandbox.open_sandbox_account(name="test")
        print("="*40)
        print(account.account_id)
        print("="*40)
        
        accounts = await client.users.get_accounts()
        for account in accounts.accounts:
            print(account)


if __name__ == "__main__":
    asyncio.run(main())
