import asyncio
from datetime import date

from src.api.coinpaprika import CoinpaprikaApi
import matplotlib.pyplot as plt


async def main():
    start = date(year=2009, month=1, day=1)
    end = date(year=2019, month=1, day=1)
    loop = asyncio.get_event_loop()
    coinpaprika = CoinpaprikaApi(loop)
    history = await coinpaprika.get_coin_history(coin_id='btc-bitcoin',
                                                 start=start,
                                                 end=end,
                                                 interval='1d')
    print(len(history))
    ohlcs = await coinpaprika.get_ohlc(coin_id='btc-bitcoin',
                                       start=start,
                                       end=end)
    print(len(ohlcs))

    events = await coinpaprika.get_events(coin_id='btc-bitcoin')
    print(events)
    print(len(events))


def plot(x, y, xlabel, ylabel, log=False):
    plt.plot(x, y)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.yscale('log' if log else 'linear')
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
