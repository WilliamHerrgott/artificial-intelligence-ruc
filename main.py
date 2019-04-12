import csv
import numpy as np
import pandas as pd

from keras.models import Sequential, Model, InputLayer
from keras.layers import LSTM, Dropout, Dense
from MyDataManager import MyDataManager

#NOTE: can we use the code we found in github??


def main():

    dm = MyDataManager("2014-01-01")
    data = dm.data_split()
    bitcoin_x = data[0]
    bitcoin_y = data[1]
    val_x = data[2]
    val_y = data[3]
    test_x = data[4]
    test_y = data[5]
    print("x: \n", bitcoin_x.shape)
    print(bitcoin_x)
    print("y: \n", bitcoin_y.shape)
    print(bitcoin_y)

    #print(bitcoin_x.shape[1])

    # Build the model
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(bitcoin_x.shape[1], 1))) # 128 -- neurons**?
    model.add(Dropout(0.2))
    model.add(Dense(units=1, activation="softmax"))  # activation function could be different
    model.compile(optimizer="adam", loss="mse")  # mse could be used for loss, look into optimiser

    model.fit(bitcoin_x, bitcoin_y,epochs=10, batch_size=8)


if __name__ == '__main__':
    main()



