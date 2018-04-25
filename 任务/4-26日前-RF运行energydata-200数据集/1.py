from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from random import randrange
from csv import reader

#Load boston housing dataset as an example
# boston = load_boston()
# X = boston["data"]
# print(X)
# Y = boston["target"]
# print(Y)
# names = boston["feature_names"]
# print(names)
# print(boston)
# print(names)

# 数据集int化

def load_csv(filename):  #导入csv文件
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# def column_to_float(dataSet):
#     featLen=len(dataSet[0])
#     for data in dataSet:
#         for column in range(featLen):
#             data[column]=float(data[column].strip())

data = 'data-200.csv'
X= load_csv(data)
target = 'target-200.csv'
Y = load_csv(target)[0]
feature = 'feature_names-10.csv'
names = load_csv(feature)[0]

# print(load_csv(data))

# 正常是一维数组
# print(load_csv(target)[0])
# print(load_csv(feature)[0])



rf = RandomForestRegressor()
rf.fit(X, Y)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),reverse=True))



# 处理数据集






