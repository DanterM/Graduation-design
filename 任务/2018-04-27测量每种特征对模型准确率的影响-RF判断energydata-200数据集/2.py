from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from random import randrange
from csv import reader
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
#
#
#
# #Load boston housing dataset as an example
# # boston = load_boston()
# # X = boston["data"]
# # print(X)
# # Y = boston["target"]
# # print(Y)
# # names = boston["feature_names"]
# # print(names)
# # print(boston)
# # print(names)
#
# # 数据集int化
#
# def load_csv(filename):  #导入csv文件
#     dataset = list()
#     with open(filename, 'r') as file:
#         csv_reader = reader(file)
#         for row in csv_reader:
#             if not row:
#                 continue
#             dataset.append(row)
#     return dataset
#
# # def column_to_float(dataSet):
# #     featLen=len(dataSet[0])
# #     for data in dataSet:
# #         for column in range(featLen):
# #             data[column]=int(data[column].strip())
#
# data = 'data-200.csv'
# X= load_csv(data)
#
# target = 'target-200.csv'
# Y = load_csv(target)[0]
# feature = 'feature_names-25.csv'
# names = load_csv(feature)[0]
#
# print(load_csv(data))
# # 二维数组到一维数组的转化
# # print(load_csv(target)[0])
# # print(load_csv(feature)[0])
#
#
#
# rf = RandomForestRegressor()
# scores = defaultdict(list)
#
# #crossvalidate the scores on a number of different random splits of the data
#
# for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
#     X_train, X_test = np.array(X)[train_idx], np.array(X)[test_idx]
#     Y_train, Y_test = np.array(Y)[train_idx], np.array(Y)[test_idx]
#     r = rf.fit(X_train, Y_train)
#     acc = r2_score(Y_test, rf.predict(X_test))
#     for i in range(X.shape[1]):
#         X_t = X_test.copy()
#         np.random.shuffle(X_t[:, i])
#         shuff_acc = r2_score(Y_test, rf.predict(X_t))
#         scores[names[i]].append((acc-shuff_acc)/acc)
# print("Features sorted by their score:")
# print(sorted([(round(np.mean(score), 4), feat) for feat, score in scores.items()], reverse=True))
#
#
# # 处理数据集
#
#
#
#
#
#








from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#Load boston housing dataset as an example
# boston = load_boston()
# X = boston["data"]
# Y = boston["target"]
# names = boston["feature_names"]


def load_csv(filename):  #导入csv文件
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

data = 'data-200.csv'
X= load_csv(data)

target = 'target-200.csv'
Y = load_csv(target)[0]
feature = 'feature_names-25.csv'
names = load_csv(feature)[0]

# print(load_csv(data))
# # 二维数组到一维数组的转化
# # print(load_csv(target)[0])
# # print(load_csv(feature)[0])

rf = RandomForestRegressor()
rf.fit(X, Y)
scores = defaultdict(list)

#crossvalidate the scores on a number of different random splits of the data
for train_idx, test_idx in ShuffleSplit(len(X), 10, .3):
    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    print(X_test)
    print(Y_test)
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    for i in range(X.shape[1]):
        X_t = X_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        print(shuff_acc)
        scores[names[i]].append((acc-shuff_acc)/acc)
print("Features sorted by their score:")
print(sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True))