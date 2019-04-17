import csv
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential, Model, InputLayer
from keras.layers import LSTM, Dropout, Dense
from MyDataManager import MyDataManager

#NOTE: can we use the code we found in github??


def main():

    dm = MyDataManager("2014-01-01")
    data = dm.data_split()
    bitcoin_x, bitcoin_y = dm.data_split()
    # val_x = data[1]
    # val_y = data[4]
    # test_y = data[3]
    # print("x: \n", bitcoin_x.shape)
    # print(bitcoin_x)
    # print("y: \n", bitcoin_y.shape)
    # print(bitcoin_y)

    #print(bitcoin_x.shape[1])

    # Build the model
    model = Sequential()
    model.add(LSTM(units=4, input_shape=(None, 1))) # 128 -- neurons**?
    # model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="softmax"))  # activation function could be different
    model.compile(optimizer="adam", loss="mean_squared_error")  # mse could be used for loss, look into optimiser

    model.fit(bitcoin_x, bitcoin_y, batch_size=32, epochs=15)

    test_set = dm.create_test_data()

    dm.plot(test_set, model)


if __name__ == '__main__':
    main()
