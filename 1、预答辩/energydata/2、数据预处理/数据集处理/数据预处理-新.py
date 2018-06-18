import numpy as np
import pandas as pd


from csv import reader
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC

def load_csv(filename): # 导入csv文件
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# 全部数据集导入!!!!!!
train_data = load_csv('data.csv')

# 三种时间处理方式

# 1-144处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1


# 秒数处理时间
# 从日期/时间变量可以生成其他额外的特性:每天从午夜开始的秒数(NSM)、周状态(周末或工作日)和一周的天数。


for i in range(len(train_data)):
    # print(X[i][0])
    chuli = train_data[i][0].split(':')
    train_data[i][0] = int(chuli[0]) * 3600 + (int(chuli[1])*60)


# print(X)

# 分钟数处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = int(chuli[0]) * 60 + (int(chuli[1]))



# 0-1分布处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = (int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1)/len(X)

print('----------data.csv时间格式已更改-----------')


# 拿出数据集训练模型X
X = train_data[:int(len(train_data)*9/10)]

# 导入结果集用于训练模型y
train_target = load_csv('target.csv')

a = []
for i in train_target:
    a.append(float(i[0]))
train_target = a
y = train_target[:int(len(train_target)*9/10)]

print('--数据集处理结束--')