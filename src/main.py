import asyncio
from datetime import date, datetime

from matplotlib.dates import YearLocator, MonthLocator, DateFormatter

from src.api.coinpaprika import CoinpaprikaApi, History
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import matplotlib.dates as mdates

register_matplotlib_converters()

from mpl_finance import candlestick_ohlc


async def main():
    start = date(year=2019, month=1, day=1)
    end = date(year=2020, month=1, day=1)

    loop = asyncio.get_event_loop()
    coinpaprika = CoinpaprikaApi(loop)

    history = await coinpaprika.get_coin_history(coin_id='btc-bitcoin',
                                                 start=start,
                                                 end=end,
                                                 interval='7d')

    ohlcs = await coinpaprika.get_ohlc(coin_id='btc-bitcoin',
                                       start=start,
                                       end=end)
    plot_candlesticks(ohlcs)
    events = await coinpaprika.get_events(coin_id='btc-bitcoin')



def plot_events(events):
    occurrences = [True for event in events]
    dates = [datetime.strptime(event.date, "%Y-%m-%dT%H:%M:%SZ") for event in events]

    plt.plot(dates, occurrences, 'b^')
    plt.ylabel('Event')
    plt.xlabel("Date")
    plt.show()


def plot_candlesticks(ohlcs):
    dates = [datetime.strptime(ohlc.time_close, "%Y-%m-%dT%H:%M:%SZ") for ohlc in ohlcs]
    opens = [ohlc.open for ohlc in ohlcs]
    closes = [ohlc.close for ohlc in ohlcs]
    highs = [ohlc.high for ohlc in ohlcs]
    lows = [ohlc.low for ohlc in ohlcs]
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)

    ax.xaxis.set_major_locator(MonthLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%Y:%m'))

    candlestick_ohlc(ax, zip(mdates.date2num(dates), opens, highs, lows, closes), width=0.6)
    ax.xaxis_date()
    ax.autoscale_view()
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    plt.show()


def plot_price(history: [History], log=False):
    prices = [hist.price for hist in history]
    dates = [datetime.strptime(hist.timestamp, "%Y-%m-%dT%H:%M:%SZ") for hist in history]

    plt.plot(dates, prices, 'b')
    plt.ylabel('Price [$]')
    plt.xlabel("Date")
    plt.yscale('log' if log else 'linear')
    plt.show()


def plot_marketcap(history: [History], log=False):
    marketcap = [hist.market_cap for hist in history]
    dates = [datetime.strptime(hist.timestamp, "%Y-%m-%dT%H:%M:%SZ") for hist in history]

    plt.plot(dates, marketcap, 'b')
    plt.ylabel('Marketcap [$]')
    plt.xlabel("Date")
    plt.yscale('log' if log else 'linear')
    plt.show()


def plot_volumes(history: [History], log=False):
    volumes = [hist.volume_24h for hist in history]
    dates = [datetime.strptime(hist.timestamp, "%Y-%m-%dT%H:%M:%SZ") for hist in history]

    plt.plot(dates, volumes, 'b')
    plt.ylabel('Volumes [$]')
    plt.xlabel("Date")
    plt.yscale('log' if log else 'linear')
    plt.show()


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_all_together(history: [History], log=False):
    volumes = [hist.volume_24h for hist in history]
    prices = [hist.price for hist in history]
    marketcap = [hist.market_cap for hist in history]
    dates = [datetime.strptime(hist.timestamp, "%Y-%m-%dT%H:%M:%SZ") for hist in history]

    fig, ax1 = plt.subplots()
    fig.subplots_adjust(right=0.75)

    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    ax3.spines['right'].set_position(('axes', 1.2))
    make_patch_spines_invisible(ax3)
    # Second, show the right spine.
    ax3.spines["right"].set_visible(True)

    ax1.set_ylabel('Price [$]', color='r')
    ax1.plot(dates, prices, 'r', label="Marketcap", alpha=0.5)
    ax1.tick_params(axis='y', labelcolor='r')

    ax2.set_ylabel('Marketcap [$]', color='b')
    ax2.plot(dates, marketcap, 'b', alpha=0.5)
    ax2.tick_params(axis='y', labelcolor='b')

    ax3.set_xlabel('Date')
    ax3.set_ylabel('Volume [$]', color='gray')
    ax3.plot(dates, volumes, label="Volume", alpha=0.3, color='gray')
    ax3.tick_params(axis='y', labelcolor='gray')

    for ax in ax1, ax2, ax3:
        ax.xaxis.set_major_locator(YearLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(MonthLocator())
        ax.autoscale_view()
        ax.set_yscale('log' if log else 'linear')

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
