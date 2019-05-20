import urllib.request
import csv
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
import os


chart_names = ["total-bitcoins", "market-price", "market-cap", "trade-volume", "blocks-size", "avg-block-size", "n-transactions-per-block", "median-confirmation-time", "hash-rate",
               "difficulty", "miners-revenue", "transaction-fees", "transaction-fees-usd", "cost-per-transaction-percent", "cost-per-transaction", "n-unique-addresses", "n-transactions",
               "n-transactions-total", "transactions-per-second", "mempool-size", "mempool-growth", "mempool-count", "utxo-count", "n-transactions-excluding-popular",
               "n-transactions-excluding-chains-longer-than-100", "output-volume", "estimated-transaction-volume-usd", "estimated-transaction-volume", "my-wallet-n-users"]

chart_names2 = ["market-price"]

def download_dataset():
    for chart_name in chart_names:
        data = urllib.request.urlopen(
            "https://api.blockchain.info/charts/" + chart_name + "?timespan=all&format=json&sampled=false").read()
        data_json = json.loads(data)
        f = csv.writer(open(chart_name + ".csv", "w", newline=''))
        x_list = []
        y_list = []
        for value in data_json['values']:
            x = value['x']
            x_list.append(x)
            y = value['y']
            y_list.append(y)
            f.writerow([value['x'], value['y']])

def build_data():
    t_set = []
    #avg_block_size = pd.read_csv('maybe/' + 'avg-block-size.csv').iloc[:, 1].values
    #difficulty = pd.read_csv('maybe/' + 'difficulty.csv').iloc[:, 1].values
    #hash_rate = pd.read_csv('maybe/' + 'hash-rate.csv').iloc[:, 1].values
    #market_cap = pd.read_csv('maybe/' + 'market-cap.csv').iloc[:, 1].values
    market_price = pd.read_csv('maybe/' + 'market-price.csv').iloc[:, 1].values # market price of bitcoin in minute interval

    for i in range(market_price.shape[0]):
        #t_set.append([avg_block_size[i], difficulty[i], hash_rate[i], market_cap[i], market_price[i]])
        t_set.append(market_price[i])

    X = np.array(t_set)
    #X = np.reshape(X, (X.shape[1], X.shape[0]))
    return X

def plot(x_list, y_list, chart_name):
    fig = plt.figure()

    ax1 = fig.add_subplot(111)

    ax1.set_title(chart_name)
    ax1.set_xlabel('timestamp')
    ax1.set_ylabel('value')

    ax1.plot(x_list, y_list, c='r', label='data')
    plt.show()

def plot(x_list, y_list, y_list2, chart_name):

    plt.plot(x_list, y_list, label='line1')
    plt.plot(x_list, y_list2, label='line2')
    plt.show()

def build_model():
    data_2018 = pd.read_csv('maybe/2018.csv', sep=';')

    total_data = pd.concat([data_2018]).iloc[:, 7].values

    #get every 10th minute
    data = total_data[0::10]

    training_set_size = int(0.9*total_data.size)
    val_set_size = training_set_size + int(0.05*total_data.size)
    test_set_size = val_set_size + int(0.05*total_data.size) + 1

    training_set = total_data[:training_set_size]
    val_set = total_data[training_set_size:val_set_size]
    test_set = total_data[val_set_size:test_set_size]

    #plot_acf(total_data)
    #pyplot.show()

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set = training_set.reshape(-1, 1)
    training_set = sc.fit_transform(training_set)

    sc2 = MinMaxScaler(feature_range=(0, 1))
    val_set = val_set.reshape(-1, 1)
    val_set = sc2.fit_transform(val_set)

    sc3 = MinMaxScaler(feature_range=(0, 1))
    test_set = test_set.reshape(-1, 1)
    test_set = sc3.fit_transform(test_set)

    X = []
    Y = []
    X_test = []
    Y_test = []

    X_val = []
    Y_val = []

    vector_size = 128

    for i in range(vector_size, int(training_set.size/8)):
        X.append(training_set[i - vector_size:i, 0])
        Y.append(training_set[i, 0])
    X, Y = np.array(X), np.array(Y)

    for i in range(vector_size, val_set.size):
        Y_val.append(val_set[i - vector_size:i, 0])
        Y_val.append(val_set[i, 0])
    X_val, Y_val = np.array(X_val), np.array(Y_val)

    for i in range(vector_size, test_set.size):
        X_test.append(test_set[i - vector_size:i, 0])
        Y_test.append(test_set[i, 0])
    X_test, Y_test = np.array(X_test), np.array(Y_test)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    regressor = Sequential()

    regressor.add(LSTM(units=64, input_shape=(X.shape[1], 1)))
    regressor.add(Dense(1))

    regressor.compile(optimizer='adagrad', loss='mse', metrics=['mse'])

    early_stopping = EarlyStopping(monitor='loss', min_delta=1)
    regressor.fit(X, Y, epochs=10, batch_size=64, callbacks=[early_stopping], validation_data=(X_val, Y_val))


    # predictions = []
    # for i in range(X_test.shape[0]):
    #     x_for_single_test = X_test[i, :, :]
    #     x_for_single_test = np.reshape(x_for_single_test, (1, 60, 1))
    #     predicted_single = regressor.predict(x_for_single_test)
    #     predicted_single = sc2.inverse_transform(predicted_single)
    #     predictions.append(predicted_single[0, 0])
    #     if i == 9999:
    #         break
    #     X_test[i + 1, 0, 0] = predicted_single

    # x_series = []
    # for i in range(0, len(predictions)):
    #     x_series.append(i)
    # plot(x_series, predictions, sc2.inverse_transform(Y_test.reshape(-1, 1)), "pred - in series")
    # plot(x_series, Y_test, "real")
    pass


    predicted = regressor.predict(X_val)
    predicted = sc2.inverse_transform(predicted)
    pass

    x_series = []
    for i in range(0, predicted.shape[0]):
        x_series.append(i)
    plot(x_series, predicted, sc2.inverse_transform(Y.reshape(-1, 1)), "pred - one shot")
    pass

#download_dataset()
build_model()
