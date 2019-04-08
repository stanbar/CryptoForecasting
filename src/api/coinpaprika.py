# https://api.coinpaprika.com/

import asyncio
import itertools
import time
from asyncio import Lock
from datetime import date, timedelta
from typing import NamedTuple

import aiohttp


class History(NamedTuple):
    timestamp: str
    price: float
    volume_24h: float
    market_cap: float


class OHLC(NamedTuple):
    time_open: str
    time_close: str
    open: float
    close: float
    high: float
    low: float
    market_cap: float


class Event(NamedTuple):
    id: str
    date: date
    date_to: date
    name: str
    description: str
    is_conference: bool
    link: str
    proof_image_link: str


class CoinpaprikaApi:

    def __init__(self, loop: asyncio.AbstractEventLoop, host='https://api.coinpaprika.com', version='v1'):
        self.loop = loop
        self.host = host
        self.lock = Lock(loop=loop)
        self.version = version
        self.request_counter = 0
        self.request_second = 0
        self.headers = {'Accept': 'application/json', 'Accept-Charset': 'utf-8'}

    async def get_coin_history(self, coin_id: str,
                               start: date = date(year=2009, month=1, day=1),  # it doesnt accept anything before
                               end: date = date.today(),
                               limit: int = 5000,
                               quote: str = "usd",
                               interval: str = '1d') -> [History]:
        return await self.with_limit(self._get_coin_history, coin_id, start, end, limit, quote, interval)

    async def _get_coin_history(self, coin_id: str, start: date, end: date, limit: int, quote: str, interval: str) -> [
        History]:
        url = f'{self.host}/{self.version}/tickers/{coin_id}/historical'
        async with aiohttp.ClientSession(headers=self.headers) as session:
            print(f'[{time.asctime()}] Send request on {url}')
            async with session.request('GET', url=url,
                                       params={
                                           'start': f'{start:%Y-%m-%d}',
                                           'end': f'{end:%Y-%m-%d}',
                                           'limit': limit,
                                           'quote': quote,
                                           'interval': interval
                                       }) as res:
                print(f'[{time.asctime()}] Receive response from {res.url}')
                print(f'[{time.asctime()}] {coin_id} fetched with status {res.status}')

                json_object = await res.json()

                if res.status == 404:  # no history found for this coin, don't raise exception
                    return []

                if type(json_object) == dict:  # unhandled exception, raise
                    raise Exception(json_object['error'])

                return [History(**json) for json in json_object]

    async def get_ohlc(self, coin_id: str,
                       start: date = date(year=2009, month=1, day=1),  # it doesnt accept anything before
                       end: date = date.today(),
                       quote: str = "usd"):

        limit = (end - start).days
        if limit <= 365:
            return await self.with_limit(self._get_ohlc, coin_id, start, end, limit, quote)
        else:
            # loop.run(None, send_batch_update, db, coins[i:i + 365])

            quotes = [await self.with_limit(self._get_ohlc, coin_id, start + timedelta(days=i),
                                            start + timedelta(days=i + 365),
                                            365,
                                            quote) for i in range(0, (end - start).days, 365)]

            return list(itertools.chain(*quotes))

    async def _get_ohlc(self, coin_id: str, start: date, end: date, limit: int, quote: str):

        url = f'{self.host}/{self.version}/coins/{coin_id}/ohlcv/historical'
        async with aiohttp.ClientSession(headers=self.headers) as session:
            print(f'[{time.asctime()}] Sending request request on {url}')
            async with session.request('GET', url=url,
                                       params={
                                           'start': f'{start:%Y-%m-%d}',
                                           'end': f'{end:%Y-%m-%d}',
                                           'limit': limit,
                                           'quote': quote
                                       }) as res:
                print(f'[{time.asctime()}] Receive response from {res.url}')
                print(f'[{time.asctime()}] {coin_id} fetched with status {res.status}')
                json_object = await res.json()

                if res.status == 404:  # no history found for this coin, don't raise exception
                    return []

                if type(json_object) == dict:  # unhandled exception, raise
                    raise Exception(json_object['error'])

                return [OHLC(**self.normalize(json)) for json in json_object]

    def normalize(self, json):
        try:
            dict.pop(json, 'volume')
        except KeyError:
            return json
        return json

    async def get_events(self, coin_id: str):
        return await self.with_limit(self._get_events, coin_id)

    async def _get_events(self, coin_id: str):
        url = f'{self.host}/{self.version}/coins/{coin_id}/events'
        async with aiohttp.ClientSession(headers=self.headers) as session:
            print(f'[{time.asctime()}] Send request on {url}')
            async with session.request('GET', url=url) as res:

                print(f'[{time.asctime()}] {coin_id} fetched with status {res.status}')
                json_object = await res.json()

                if res.status == 404:  # no history found for this coin, don't raise exception
                    return []

                if type(json_object) == dict:  # unhandled exception, raise
                    raise Exception(json_object['error'])

                return [Event(**json) for json in json_object]

    async def with_limit(self, func, *args, **kwargs):
        task = None
        await self.lock.acquire()
        try:
            if self.request_second == int(time.time()):
                if self.request_counter == 9:
                    print(f'[{time.asctime()}] limit reached sleeping 1 sec')
                    await asyncio.sleep(1, loop=self.loop)
                    self.request_second = int(time.time())
                    self.request_counter = 1
                else:
                    self.request_counter += 1
                task = func(*args, **kwargs)
            else:
                self.request_second = int(time.time())
                self.request_counter = 1
                task = func(*args, **kwargs)
        finally:
            result = await task
            self.lock.release()
            return result
