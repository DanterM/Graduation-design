import csv
from random import seed
from random import randrange
from math import sqrt

from sklearn import datasets

iris = datasets.load_iris()
print(iris)

X = iris.data
# print(X)
y = iris.target
# print(y)


def loadCSV(filename):
    dataSet = []
    with open(filename, 'r') as file:
        csvReader = csv.reader(file)
        for line in csvReader:
            dataSet.append(line)
    return dataSet



dataSet = loadCSV('energydata_complete.csv')
# print(dataSet)
