import csv
import pandas as pd
import numpy as np

from keras.models import Sequential, Model, InputLayer
from keras.layers import Dense


# change the name of this file to your own path
fileName = "bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"

with open(fileName, 'r') as csvFile:
    data = csv.reader(csvFile)

csvFile.close()
