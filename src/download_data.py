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
from keras.losses import mean_squared_error
from keras.callbacks import EarlyStopping
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib import pyplot
import os

chart_names = ["total-bitcoins", "market-price", "market-cap", "trade-volume", "blocks-size", "avg-block-size",
               "n-transactions-per-block", "median-confirmation-time", "hash-rate",
               "difficulty", "miners-revenue", "transaction-fees", "transaction-fees-usd",
               "cost-per-transaction-percent", "cost-per-transaction", "n-unique-addresses", "n-transactions",
               "n-transactions-total", "transactions-per-second", "mempool-size", "mempool-growth", "mempool-count",
               "utxo-count", "n-transactions-excluding-popular",
               "n-transactions-excluding-chains-longer-than-100", "output-volume", "estimated-transaction-volume-usd",
               "estimated-transaction-volume", "my-wallet-n-users"]

chart_names2 = ["market-price"]


def plot(x_list, y_list, y_list2, chart_name):
    plt.plot(x_list, y_list, label='line1')
    plt.plot(x_list, y_list2, label='line2')
    plt.title(chart_name)
    plt.legend(['predicted', 'real'], loc='upper left')
    plt.show()


def create_model():
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(rate=0.3))
    model.add(Dense(1))
    model.compile(optimizer='adagrad', loss='mse')
    return model


def train_model(model):
    story = model.fit(x_train, y_train, epochs=20, batch_size=128, callbacks=[], validation_data=(x_val, y_val))
    plt.plot(story.history['loss'])
    plt.plot(story.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    model.save('MODEL_SPLIT1.h5')


normalization = True
data_2018 = pd.read_csv('bitcoin_market_data.csv', sep=',')

total_data = pd.concat([data_2018]).iloc[:, 2:5].values

train_set_size = int(0.9 * total_data.shape[0])
val_set_size = train_set_size + int(0.05 * total_data.shape[0])
test_set_size = val_set_size + int(0.05 * total_data.shape[0]) + 1

train_set = total_data[:train_set_size, :]
val_set = total_data[train_set_size:val_set_size, :]
test_set = total_data[val_set_size:test_set_size, :]

train_scaller = MinMaxScaler(feature_range=(0, 1))
if normalization:
    train_set = train_set.reshape(-1, 1)
    train_set = train_scaller.fit_transform(train_set)

val_scaller = MinMaxScaler(feature_range=(0, 1))
if normalization:
    val_set = val_set.reshape(-1, 1)
    val_set = val_scaller.fit_transform(val_set)

test_scaller = MinMaxScaler(feature_range=(0, 1))
if normalization:
    test_set = test_set.reshape(-1, 1)
test_set2 = test_set.copy()[:, 0]
if normalization:
    test_set = test_scaller.fit_transform(test_set)

test_scaller2 = MinMaxScaler(feature_range=(0, 1))
test_set2 = test_set2.reshape(-1, 1)
if normalization:
    test_set2 = test_scaller2.fit_transform(test_set2)

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []

vector_size = 128
for i in range(vector_size, train_set.shape[0]):
    x_train.append(train_set[i - vector_size:i, :])
    y_train.append(train_set[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
for i in range(vector_size, val_set.shape[0]):
    x_val.append(val_set[i - vector_size:i, :])
    y_val.append(val_set[i, 0])

x_val, y_val = np.array(x_val), np.array(y_val)

for i in range(vector_size, test_set.shape[0]):
    x_test.append(test_set[i - vector_size:i, :])
    y_test.append(test_set[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2]))  # todo tuple index out of range
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

model = create_model()

# train_model(model)
model.load_weights('MODEL_SPLIT3.h5')
predicted = model.predict(x_val)
if normalization:
    predicted = test_scaller2.inverse_transform(predicted)

evaluate = model.evaluate(x=x_val, y=y_val)

print(evaluate)

x_series = list(range(0, predicted.shape[0]))
x_series = np.reshape(x_series, (x_series.__len__(), 1))
if normalization:
    plot(x_series, predicted, test_scaller2.inverse_transform(y_val.reshape(-1, 1)), "pred - one shot")
else:
    plot(x_series, predicted, y_val.reshape(-1, 1), "pred - one shot")
