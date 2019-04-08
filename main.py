import csv

from keras.models import Sequential, Model, InputLayer
from keras.layers import Dense

fileName = "bitcoin-historical-data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv"

with open(fileName, 'r') as csvFile:
    data = csv.reader(csvFile)

csvFile.close()
