import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cryptory import Cryptory
from sklearn.preprocessing import MinMaxScaler


class MyDataManager():

    def __init__(self, date):
        self.date = date
        self.crypto_data = Cryptory(from_date = self.date)
        self.bitcoin_data = self.crypto_data.extract_coinmarketcap("bitcoin")
        self.sc = MinMaxScaler()


    # Cleans the data to keep only the open column
    def clean_data(self):
        for col in self.bitcoin_data.columns:
            if col != "open":
                del self.bitcoin_data[col]


    # Creates the train data set
    def create_train_data(self):

        self.clean_data()

        training_set = self.bitcoin_data;
        training_set = self.sc.fit_transform(training_set)
        train_data = training_set[365:]

        # Split the data into x and y
        x_train, y_train = train_data[:len(train_data)-1], train_data[1:]

        return self.format_to_3d(x_train), y_train


    # Creates the test data set
    def create_test_data(self):
        test_set = self.bitcoin_data
        test_set = self.sc.transform(test_set)
        test_data = test_set[:364]

        return test_data


    # Plots the graph
    def plot(self, real_value, model):
        input = real_value
        input = self.sc.fit_transform(input)
        input = np.reshape(input, (364, 1, 1))

        predicted_result = model.predict(input)
        predicted_result = self.sc.inverse_transform(predicted_result)

        plt.plot(real_value, color='pink', label='Real Price')
        plt.plot(predicted_result, color='blue', label='Predicted Price')
        plt.title('Bitcoin Prediction')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        plt.legend()
        plt.show()


    # Format a 2D np.array to a 3D array
    def format_to_3d(self, df_to_reshape):
        reshaped_df = np.array(df_to_reshape)
        return np.reshape(reshaped_df, (reshaped_df.shape[0], 1, reshaped_df.shape[1]))
