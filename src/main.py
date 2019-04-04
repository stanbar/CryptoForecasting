import asyncio
from datetime import date

from src.api.coinpaprika import CoinpaprikaApi


async def main():
    print('main function')
    loop = asyncio.get_event_loop()
    coinpaprika = CoinpaprikaApi(loop)
    history = await coinpaprika.get_coin_history(coin_id='btc-bitcoin')
    for day in history:
        print(day)

    ohlcs = await coinpaprika.get_ohlc(coin_id='btc-bitcoin',
                                       start=date(year=2017, month=1, day=1),
                                       end=date(year=2018, month=1, day=1))
    for ohlc in ohlcs:
        print(ohlc)

    events = await coinpaprika.get_events(coin_id='btc-bitcoin')
    for event in events:
        print(event)


if __name__ == "__main__":
    asyncio.run(main())
