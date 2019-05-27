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

data_2018 = pd.read_csv('maybe/2018.csv', sep=';')

total_data = pd.concat([data_2018]).iloc[:, 7].values

#get every 10th minute
total_data = total_data[0::10]

train_set_size = int(0.9*total_data.size)
val_set_size = train_set_size + int(0.05*total_data.size)
test_set_size = val_set_size + int(0.05*total_data.size) + 1

train_set = total_data[:val_set_size]
val_set = total_data[train_set_size:val_set_size]
test_set = total_data[val_set_size:test_set_size]

train_scaller = MinMaxScaler(feature_range=(0, 1))
train_set = train_set.reshape(-1, 1)
train_set = train_scaller.fit_transform(train_set)

val_scaller = MinMaxScaler(feature_range=(0, 1))
val_set = val_set.reshape(-1, 1)
val_set = val_scaller.fit_transform(val_set)

test_scaller = MinMaxScaler(feature_range=(0, 1))
test_set = test_set.reshape(-1, 1)
test_set = test_scaller.fit_transform(test_set)

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

vector_size = 512

for i in range(vector_size, train_set.size):
    x_train.append(train_set[i - vector_size:i, 0])
    y_train.append(train_set[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

for i in range(vector_size, val_set.size):
    x_val.append(val_set[i - vector_size:i, 0])
    y_val.append(val_set[i, 0])
x_val, y_val = np.array(x_val), np.array(y_val)

for i in range(vector_size, test_set.size):
    x_test.append(test_set[i - vector_size:i, 0])
    y_test.append(test_set[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

regressor = Sequential()

regressor.add(LSTM(units=64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
regressor.add(LSTM(units=64))
regressor.add(Dropout(rate=0.2))
regressor.add(Dense(1))

regressor.compile(optimizer='adagrad', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5)
regressor.fit(x_train, y_train, epochs=4, batch_size=64, callbacks=[early_stopping], validation_data=(x_val, y_val))


predicted = regressor.predict(x_test)
predicted = test_scaller.inverse_transform(predicted)

x_series = list(range(0, predicted.shape[0]))
x_series = np.reshape(x_series, (x_series.__len__(), 1))
plot(x_series, predicted, test_scaller.inverse_transform(y_test.reshape(-1, 1)), "pred - one shot")
pass
