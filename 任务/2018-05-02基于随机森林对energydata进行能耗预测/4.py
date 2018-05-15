#!/usr/bin/env python
# coding=utf8

import sys
# reload(sys)
# sys.setdefaultencoding('utf8')

'''
Competition URL: https://www.kaggle.com/c/digit-recognizer
Solution:  Random Forest
'''


# 引入需要的包
# 数据处理的常用包
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import tree
# 随机森林的包
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
import pydotplus
from IPython.display import Image
# 画图的包
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
import math

# 读取数据（请先去 https://www.kaggle.com/c/digit-recognizer/data 上下载数据）
# 读取成DataFrame的数据
train_df = pd.read_csv('train.csv')
# 将DataFrame的数据转换成Array
train_data = train_df.values

test_df = pd.read_csv('test.csv')
test_data = test_df.values

correctAnswer = pd.read_csv('correctAnswer.csv')
# correctAnswer = correctAnswer.astype('float64')



# print(train_df.head())
# print(train_data)
# print(correctAnswer)


# 画图
plt.figure(figsize=(12,8)).show()
sns.countplot(x='Appliances', data=train_df)
plt.title('Distribution of Numbers')
plt.xlabel('Numbers');
plt.show()

# 2/3的train_data作为训练数据，1/3的train_data作为测试数据来训练模型
num_features = train_data.shape[0] # 拿到train_data的行数，也就是所有数据的个数作为特征值
print("Number of all features: \t\t", num_features)
split = int(num_features * 2/3) # 这里是取2/3行作为训练 后1/3行作为测试

train = train_data[:split] # 取出前2/3行作为训练数据
test = train_data[split:] # 取出后1/3行作为测试数据

print("Number of features used for training: \t", len(train),
      "\nNumber of features used for testing: \t", len(test))

# 开始使用随机森林分类器
clf = RandomForestClassifier(n_estimators=200) # 定义决策树的个数为200

# 开始训练，训练的X数据格式为[[]]，训练的y值为[]也就是经过ravel后的数据
# 如果你问我ravel()的作用是什么，就是不管什么数据格式的数据都转成一个一维的array，这样每个元素都是一个平等且顺序的位置
model = clf.fit(train[:,1:], train[:,0].ravel())

# 然后预测
output = model.predict(test[:,1:])
print(output)
pd.DataFrame({"Id": range(1, len(output)+1), "Label": output}).to_csv('out0.csv', index=False, header=True)




# 计算准确度
acc = np.mean(output == test[:,0].ravel()) *100


print("The accuracy of the pure RandomForest classifier is: \t", acc, "%")


# 利用
clf = RandomForestClassifier(n_estimators=100) # 100 trees

# 用全部训练数据来做训练
target = train_data[:,0].ravel()
train = train_data[:,1:]
model = clf.fit(train, target)




# 用测试集数据来预测最终结果
output = model.predict(test_data)

print(output)


# 计算误差率
# for j in correctAnswer:
#       print(j)
#       for i in output1:
#             print(i)
#             accrucy = abs(int((i-j)/j))
#             print(accrucy)




# 均方根误差（RMSE）
def rmse(y_test, y):
    return sp.sqrt(sp.mean((y_test - y) ** 2))

# 输出预测结果
pd.DataFrame({"Id": range(1, len(output)+1), "Label": output}).to_csv('out.csv', index=False, header=True)
