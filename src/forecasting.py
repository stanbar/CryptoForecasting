import asyncio

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from src.api.coinpaprika import CoinpaprikaApi

# using https://www.datacamp.com/community/tutorials/lstm-python-stock-market
async def main():
    file_to_save = 'bitcoin_market_data.csv'
    if not os.path.exists(file_to_save):
        loop = asyncio.get_event_loop()
        coinpaprika = CoinpaprikaApi(loop)
        start = dt.date(year=2009, month=1, day=1)
        end = dt.date(year=2020, month=1, day=1)
        ohlcs = await coinpaprika.get_ohlc(coin_id='btc-bitcoin',
                                           start=start,
                                           end=end)
        df = pd.DataFrame(columns=['Date', 'Low', 'High', 'Close', 'Open'])
        for ohlc in ohlcs:
            date = dt.datetime.strptime(ohlc.time_close, '%Y-%m-%dT%H:%M:%SZ')
            data_row = [date.date(), ohlc.low, ohlc.high, ohlc.close, ohlc.open]
            df.loc[-1, :] = data_row
            df.index = df.index + 1

        print(f'Data saved to : {file_to_save}')
        df.to_csv(file_to_save)
    else:
        print('File already exists. Loading data from CSV')
        df = pd.read_csv(file_to_save)

    # Sort DataFrame by date
    df = df.sort_values('Date')
    # Double check the result
    df.head()

    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), (df['Low'] + df['High']) / 2.0)
    plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Mid Price', fontsize=18)
    plt.show()

    high_prices = df.loc[:, 'High'].as_matrix()
    low_prices = df.loc[:, 'Low'].as_matrix()
    mid_prices = (high_prices + low_prices) / 2.0
    train_data = mid_prices[:2000]
    test_data = mid_prices[2000:]

    # scale the data to be between 0 and 1
    # When scaling remember! You normalize both test and train data with respect to training data
    # Because you are not supposed to have access to test data
    scaler = MinMaxScaler()
    train_data = train_data.reshape(-1, 1)
    test_data = test_data.reshape(-1, 1)

    # Train the Scaler with training data and smooth data
    smoothing_window_size = 500
    for di in range(0, train_data.size, smoothing_window_size):
        scaler.fit(train_data[di:di + smoothing_window_size, :])
        train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

    # Reshape both train and test data
    train_data = train_data.reshape(-1)
    # Normalize test data
    test_data = scaler.transform(test_data).reshape(-1)

    # Now perform exponential moving average smoothing
    # So the data will have a smoother curve than the original ragged data
    EMA = 0.0
    gamma = 0.1
    for ti in range(train_data.size):
        EMA = gamma * train_data[ti] + (1 - gamma) * EMA
        train_data[ti] = EMA

    # Used for visualization and test purposes
    all_mid_data = np.concatenate([train_data, test_data], axis=0)

    # Standard Average
    window_size = 20
    N = train_data.size
    std_avg_predictions = []
    std_avg_x = []
    mse_errors = []

    for pred_idx in range(window_size, N):

        if pred_idx >= N:
            date = dt.datetime.strptime(df['Date'][pred_idx], '%Y-%m-%d').date() + dt.timedelta(days=1)
        else:
            date = df.loc[pred_idx, 'Date']

        std_avg_predictions.append(np.mean(train_data[pred_idx - window_size:pred_idx]))
        mse_errors.append((std_avg_predictions[-1] - train_data[pred_idx]) ** 2)
        std_avg_x.append(date)

    print('MSE error for standard averaging: %.5f' % (0.5 * np.mean(mse_errors)))

    plt.figure(figsize=(18, 9))
    plt.plot(range(df.shape[0]), all_mid_data, color='b', label='True')
    plt.plot(range(window_size, N), std_avg_predictions, color='orange', label='Prediction')
    plt.xlabel('Date')
    plt.ylabel('Mid Price')
    plt.legend(fontsize=18)
    plt.show()


if __name__ == "__main__":
    asyncio.run(main())
