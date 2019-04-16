import numpy as np
import pandas as pd

from cryptory import Cryptory
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class MyDataManager():

    def __init__(self, date):
        self.date = date
        self.crypto_data = Cryptory(from_date = self.date)
        self.bitcoin_data = self.crypto_data.extract_coinmarketcap("bitcoin")
        self.sc = MinMaxScaler()

    def clean_data(self):
        for col in self.bitcoin_data.columns:
            if col != "open":
                del self.bitcoin_data[col]


    #TODO figure out what to do with windows
    #def create_window(self, data, window_length):
        # Create batches with the given sequence length
     #   data_with_window = []
     #   for i in range(len(data)-window_length+1):
     #       data_with_window.append(data[i:i+window_length])

      #  return data_with_window


    #TODO normalise the data
    def normalise(self):
        self.bitcoin_data = self.sc.fit_transform(self.bitcoin_data)


    def data_split(self,train_size=0.6, test_size=0.2): # just followed the example

        self.clean_data()
        self.normalise()
        data = np.array(self.bitcoin_data)

        # Calculate the splitting indices
        train_split = int(round(train_size * data.shape[0]))
        test_split = int(train_split + round(test_size * data.shape[0]))

        # Split the data into train, validate and test
        train_data = data[:train_split]
        validate_data = data[train_split:test_split]
        test_data = data[test_split:]

        # Split the data into x and y
        x_train, y_train = train_data[:len(train_data)-1], train_data[1:]
        #x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        #y_train = np.reshape(y_train, (y_train.shape[0], y_train.shape[1], 1))
        x_validate, y_validate = validate_data[:len(validate_data)-1], validate_data[1:]
        x_test, y_test = test_data[:len(train_data)-1], test_data[1:]

        result = self.format_to_3d([x_train, x_validate, x_test])
        result += [y_train, y_validate, y_test]

        return result


    def format_to_3d(self, df_to_reshape):

        if (isinstance(df_to_reshape, list)):
            for i in range(len(df_to_reshape)):
                reshaped_df = np.array(df_to_reshape[i])
                df_to_reshape[i] = np.reshape(reshaped_df, (reshaped_df.shape[0], 1, reshaped_df.shape[1]))
            return df_to_reshape
        else:
            reshaped_df = np.array(df_to_reshape)
            return np.reshape(reshaped_df, (reshaped_df.shape[0], 1, reshaped_df.shape[1]))


    def plot(self, predicted_result, real_value):
        predicted_result = self.sc.inverse_transform(predicted_result)
        plt.plot(real_value, color='pink', label='Real Price')
        plt.plot(predicted_result, color='blue', label='Predicted Price')
        plt.title('Bitcoin Prediction')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        plt.legend()
        plt.show()
