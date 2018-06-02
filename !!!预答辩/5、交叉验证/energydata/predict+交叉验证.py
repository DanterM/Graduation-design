# 加上时间特征之后数据的格式发生变化 Appliances不是第一列了 ？？？可能会产生问题 已解决

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # 分割数据模块
from csv import reader
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def load_csv(filename):  #导入csv文件
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset



# 200个数据集导入
# X= load_csv('data-200.csv')
# print(X)
#
# y = load_csv('target-200.csv')[0]
# print(y)



# 全部数据集导入
X= load_csv('data.csv')
# print(len(X))
# print(type(X[0][0]))
print('----------data.csv导入成功----------')

y = load_csv('target.csv')
print('----------target.csv导入成功----------')
# 处理结果target数组 得到y
a = []
for i in y:
    # print(i)
    a.append(float(i[0]))
    # a.append(i[0])

y = a
print('----------target.csv处理完成----------')

# print(y)








# 三种时间处理方式

# 1-144处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1


# 秒数处理时间
# 从日期/时间变量可以生成其他额外的特性:每天从午夜开始的秒数(NSM)、周状态(周末或工作日)和一周的天数。


for i in range(len(X)):
    # print(X[i][0])
    chuli = X[i][0].split(':')
    X[i][0] = int(chuli[0]) * 3600 + (int(chuli[1])*60)
print('----------data.csv处理完成-时间格式已更改----------')

# print(X)


# 分钟数处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = int(chuli[0]) * 60 + (int(chuli[1]))



# 0-1分布处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = (int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1)/len(X)

from sklearn.neighbors import KNeighborsClassifier  # K最近邻(kNN，k-NearestNeighbor)分类算法



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# randomForest = RandomForestClassifier()
# randomForest = RandomForestRegressor()
# randomForest = DecisionTreeClassifier()
# randomForest = DecisionTreeRegressor()
knn = KNeighborsClassifier()
# knn = SVC()

knn.fit(X_train, y_train)
# randomForest.fit(X_train,y_train)
# print('正确率为：',randomForest.score(X_test, y_test))
print('正确率为：',knn.score(X_test, y_test))


from sklearn.cross_validation import cross_val_score  # K折交叉验证模块
from sklearn.cross_validation import KFold

# 使用K折交叉验证模块
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
# scores = cross_val_score(randomForest, X, y, cv=10, scoring='accuracy')
# scores = cross_val_score(randomForest, X, y, cv=10, scoring='r2') #重复cv次交叉验证
# scores = cross_val_score(randomForest, X, y, cv=10, scoring='precision')
KFold(10, n_folds=2)


# 将5次的预测准确率打印出
print(scores)
# [ 0.96666667  1.          0.93333333  0.96666667  1.        ]

# 将5次的预测准确平均率打印出
print(scores.mean())



import matplotlib.pyplot as plt  # 可视化模块

# 建立测试参数集
k_range = range(1, 100)

k_scores = []

# 藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # random = RandomForestRegressor(n_estimators=k)

    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    # scores = cross_val_score(random, X, y, cv=10, scoring='accuracy')

    k_scores.append((scores).mean())


# 可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for RandomForest')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
