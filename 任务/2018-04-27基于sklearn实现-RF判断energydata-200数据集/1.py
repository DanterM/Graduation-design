from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from random import randrange
from csv import reader



# 基于不纯度对模型进行排序有几点需要注意：
# （1）基于不纯度降低的特征选择将会偏向于选择那些具有较多类别的变量（bias）。
# （2）当存在相关特征时，一个特征被选择后，与其相关的其他特征的重要度则会变得很低，因为他们可以减少的不纯度已经被前面的特征移除了。

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
# print(X)
target = 'target-200.csv'

Y = load_csv(target)[0]
# print(Y)
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



