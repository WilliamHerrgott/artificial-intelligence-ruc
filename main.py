import csv
import random
import numpy as np
import pandas as pd

from cryptory import Cryptory
from keras.models import Sequential, Model, InputLayer
from keras.layers import LSTM, Dropout, Dense

#NOTE: can we use the code we found in github??

def create_window(data, window_length):
    # Create batches with the given sequence length
    data_with_window = []
    for i in range(len(data)-window_length+1):
        data_with_window.append(data[i:i+window_length])

    return data_with_window

def normalise(window):
    # Normalise the given window
    ...

def data_split(data, train_size=0.6, test_size=0.2): # just followed the example

    data = np.array(data)
    # Calculate the splitting indices
    train_split = round(train_size * data.shape[0])
    test_split = train_split + round(test_size * data.shape[0])
    # Shuffle and Split the data into train, validate and test
    random.shuffle(data)   # ARE WE GOING TO SHUFFLE -- IN ALL OF THE EXAMPLES I SAW THERE WERE NO SHUFFLING
    train_data = data[:train_split]
    validate_data = data[train_split:test_split]
    test_data = data[test_split:]

    # Split the data into x and y
    x_train, y_train = train_data[:train_split, :-1], train_data[:train_split, -1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    #x_validate, y_validate
    #x_test, y_test = test_data[test_split:, :-1], test_data[test_split:, -1]   NOT SURE

    return x_train, y_train


def format_to_lstm(reshaped_df):
    #reshaped_df = np.array(df_to_reshape)
    return np.reshape(reshaped_df, (reshaped_df.shape[0], 1, reshaped_df.shape[1]))

def main():
    # Get the data and split it into 3 parts
    crypto_data = Cryptory(from_date = "2014-01-01")
    bitcoin_data = crypto_data.extract_coinmarketcap("bitcoin")

    # Delete the unnecessary columns
    for col in ["high", "low", "marketcap"]:
        del bitcoin_data[col]

    print(bitcoin_data)
    bitcoin_x, bitcoin_y = data_split(bitcoin_data)
    print("x: \n", bitcoin_x)
    print("y: \n", bitcoin_y)



    # bitcoin_data is a pandas Data Frame object
    #bitcoin_data = format_to_lstm(bitcoin_data)  # now its a numpy array
    # The total number of data: 1925 -- randomly split this into train(1165), validate(380), test(380)
    #random.shuffle(bitcoin_data)
    #train_data = bitcoin_data[:1165]
    #validate_data = bitcoin_data[1165:1545]
    #test_data = bitcoin_data[1545:]

    #print(format_to_lstm(bitcoin_data))

    #print("Shape of train data: ", train_data.shape, train_data.ndim)
    #print("Shape of validate data: ", validate_data.shape)
    #print("Shape of test data: ", test_data.shape)

if __name__ == '__main__':
    main()



# Build the model
#model = Sequential()
#model.add(LSTM(128), input_shape=...)  # 128 -- neurons ;
#model.add(Dropout(0.25))
#model.add(Dense(units=..., activation="softmax"))  # activation function could be different
#model.compile(loss="mae", optimizer="adam")  # mse could be used for loss, look into optimiser
