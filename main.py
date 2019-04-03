import csv

fileName = ""

with open(fileName, 'r') as csvFile:
    data = csv.reader(csvFile)

csvFile.close()