import matplotlib.pyplot as plt  # 可视化模块
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


X= load_csv('data.csv')
for i in range(len(X)):
    # print(X[i][0])
    chuli = X[i][0].split(':')
    X[i][0] = int(chuli[0]) * 3600 + (int(chuli[1])*60)
print('----------data.csv处理完成-时间格式已更改----------')



y = load_csv('target.csv')
print('----------target.csv导入成功----------')
# 处理结果target数组 得到y
a = []
for i in y:
    # print(i)
    a.append(float(i[0]))
    # a.append(i[0])

y = a



# 建立测试参数集
k_range = range(1, 101, 10)
k_scores = []
for k in k_range:
    random = SVC()
    random.fit(X, y)
    predict = random.predict(X)
    a = 0
    wucha = 10
    for i in range(len(y)):
        if abs((predict - y)[i]) < wucha:
            a += 1
    acc = a / len(y)
    k_scores.append(acc)
# 可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for svm')
plt.ylabel('Accuracy')
plt.show()
