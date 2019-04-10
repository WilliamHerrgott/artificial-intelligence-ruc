import csv
import pandas as pd
import numpy as np
import random

from keras.models import Sequential, Model, InputLayer
from keras.layers import LSTM, Dropout, Dense


from cryptory import Cryptory

# Get the data and split it into 3 parts
crypto_data = Cryptory(from_date = "2014-01-01")
bitcoin_data = crypto_data.extract_coinmarketcap("bitcoin")  # bitcoin_data is a pandas Data Frame object
bitcoin_data = np.array(bitcoin_data)  # now its a numpy array
# The total number of data: 1925 -- randomly split this into train(1165), validate(380), test(380)
random.shuffle(bitcoin_data)
train_data = bitcoin_data[:1165]
validate_data = bitcoin_data[1165:1545]
test_data = bitcoin_data[1545:]

print("Shape of train data: ", train_data.shape)
print("Shape of validate data: ", validate_data.shape)
print("Shape of test data: ", test_data.shape)


# Build the model
#model = Sequential()
#model.add(LSTM(128), input_shape=...)  # 128 -- neurons ;
#model.add(Dropout(0.25))
#model.add(Dense(units=..., activation="softmax"))  # activation function could be different
#model.compile(loss="mae", optimizer="adam")  # mse could be used for loss, look into optimiser
