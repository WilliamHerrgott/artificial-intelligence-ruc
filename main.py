import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cryptory import Cryptory
from keras.models import Sequential, Model, InputLayer
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler


def format_to_3d(df_to_reshape):
    reshaped_df = np.array(df_to_reshape)
    return np.reshape(reshaped_df, (reshaped_df.shape[0], 1, reshaped_df.shape[1]))


crypto_data = Cryptory(from_date = "2014-01-01")
bitcoin_data = crypto_data.extract_coinmarketcap("bitcoin")

sc = MinMaxScaler()

for col in bitcoin_data.columns:
    if col != "open":
        del bitcoin_data[col]

training_set = bitcoin_data;
training_set = sc.fit_transform(training_set)

# Split the data into train, validate and test
train_data = training_set[365:]

# Split the data into x and y
x_train, y_train = train_data[:len(train_data)-1], train_data[1:]

model = Sequential()
model.add(LSTM(units=4, input_shape=(None, 1))) # 128 -- neurons**?
# model.add(Dropout(0.2))
model.add(Dense(units=1, activation="softmax"))  # activation function could be different
model.compile(optimizer="adam", loss="mean_squared_error")  # mse could be used for loss, look into optimiser

model.fit(format_to_3d(x_train), y_train, batch_size=32, epochs=15)

test_set = bitcoin_data
test_set = sc.transform(test_set)
test_data = test_set[:364]

input = test_data
input = sc.inverse_transform(input)
input = np.reshape(input, (364, 1, 1))

predicted_result = model.predict(input)
print(predicted_result)

real_value = sc.inverse_transform(test_data)

plt.plot(real_value, color='pink', label='Real Price')
plt.plot(predicted_result, color='blue', label='Predicted Price')
plt.title('Bitcoin Prediction')
plt.xlabel('Time')
plt.ylabel('Prices')
plt.legend()
plt.show()

# dm.plot(test_set, model)
