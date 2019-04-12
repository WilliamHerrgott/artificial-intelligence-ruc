import csv
import numpy as np
import pandas as pd

from keras.models import Sequential, Model, InputLayer
from keras.layers import LSTM, Dropout, Dense
from MyDataManager import MyDataManager

#NOTE: can we use the code we found in github??

# WE'RE GOING TO USE THE OPEN VALUES AS INPUT(X), THE Y WILL BE THE "NEXT" VALUE



def main():

    dm = MyDataManager("2014-01-01")
    data = dm.data_split()
    print(data[0].shape)
    #bitcoin_x, bitcoin_y, val_x, val_y, test_x, test_y = dm.data_split()
    #print("x: \n", bitcoin_x.shape)
    #print("y: \n", bitcoin_y.shape)





    #bitcoin_x, bitcoin_y = data_split(bitcoin_data)
    #print("x: \n", bitcoin_x)
    #print("y: \n", bitcoin_y)



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
