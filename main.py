import csv
import pandas as pd
import numpy as np

from keras.models import Sequential, Model, InputLayer
from keras.layers import Dense

from cryptory import Cryptory

crypto_data = Cryptory(from_date = "2014-01-01")
bitcoin_data = crypto_data.extract_coinmarketcap("bitcoin")
# The total number of data: 1925 -- randomly split this into train(1165), validate(380), test(380)
count_row = bitcoin_data.shape[0]
print(count_row)
