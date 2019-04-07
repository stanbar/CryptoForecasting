import asyncio
from datetime import date

from src.api.coinpaprika import CoinpaprikaApi


async def main():
    start = date(year=2017, month=1, day=1)
    end = date(year=2018, month=1, day=1)
    loop = asyncio.get_event_loop()
    coinpaprika = CoinpaprikaApi(loop)
    history = await coinpaprika.get_coin_history(coin_id='btc-bitcoin')
    print(history)

    ohlcs = await coinpaprika.get_ohlc(coin_id='btc-bitcoin',
                                       start=start,
                                       end=end)
    print(ohlcs)

    events = await coinpaprika.get_events(coin_id='btc-bitcoin')
    print(events)


if __name__ == "__main__":
    asyncio.run(main())
