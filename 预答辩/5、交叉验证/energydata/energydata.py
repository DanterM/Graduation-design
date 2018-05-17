from sklearn import cross_validation
import numpy as np
import pandas as pd
from csv import reader
from sklearn.ensemble import RandomForestRegressor



def load_csv(filename):  #导入csv文件
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset



# data数据集导入
X= load_csv('data.csv')

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


for i in range(len(X)):
    # print(X[i][0])
    chuli = X[i][0].split(':')
    X[i][0] = int(chuli[0]) * 3600 + (int(chuli[1])*60)
print('----------data.csv处理完成-时间格式已更改----------')



X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
print(X_train.shape)


# clf = RandomForestRegressor.fit(X_train,y_train,y)
# clf.score(X_test, y_test)