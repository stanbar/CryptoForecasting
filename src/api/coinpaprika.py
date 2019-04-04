import asyncio
import time
from asyncio import Lock
from datetime import date

import aiohttp


class CoinpaprikaApi:

    def __init__(self, loop, host='https://api.coinpaprika.com', version='v1'):
        self.loop = loop
        self.host = host
        self.lock = Lock(loop=loop)
        self.version = version
        self.request_counter = 0
        self.request_second = 0
        self.headers = {'Accept': 'application/json', 'Accept-Charset': 'utf-8'}

    async def get_ohlc(self, *args, **kwargs):
        return await self.with_limit(self._get_ohlc, *args, **kwargs)

    async def _get_ohlc(self,
                        coin_id: str,
                        start: date = date(year=2009, month=1, day=1),  # it doesnt accept anything before
                        end: date = date.today(),
                        limit: int = 365,
                        quote: str = "usd"):
        url = f'{self.host}/{self.version}/coins/{coin_id}/ohlcv/historical'
        async with aiohttp.ClientSession(headers=self.headers) as session:
            print(f'[{time.asctime()}] Send request on {url}')
            async with session.request('GET', url=url,
                                       params={
                                           'start': f'{start:%Y-%m-%d}',
                                           'end': f'{end:%Y-%m-%d}',
                                           'limit': limit,
                                           'quote': quote
                                       }) as res:

                print(f'[{time.asctime()}] {coin_id} fetched with status {res.status}')
                json_object = await res.json()

                if res.status == 404:  # no history found for this coin, don't raise exception
                    return []

                if type(json_object) == dict:  # unhandled exception, raise
                    raise Exception(json_object['error'])

                return json_object

    async def get_coin_history(self, *args, **kwargs):
        return await self.with_limit(self._get_coin_history, *args, **kwargs)

    async def _get_coin_history(self,
                                coin_id: str,
                                start: date = date(year=2009, month=1, day=1),  # it doesnt accept anything before
                                end: date = date.today(),
                                limit: int = 5000,
                                quote: str = "usd",
                                interval: str = '1d'):
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

                print(f'[{time.asctime()}] {coin_id} fetched with status {res.status}')
                json_object = await res.json()

                if res.status == 404:  # no history found for this coin, don't raise exception
                    return []

                if type(json_object) == dict:  # unhandled exception, raise
                    raise Exception(json_object['error'])

                return json_object

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
