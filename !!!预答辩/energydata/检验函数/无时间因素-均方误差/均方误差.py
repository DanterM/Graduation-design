# 引入需要的包
# 数据处理的常用包
import numpy as np
import pandas as pd
import scipy as sp
from sklearn import tree
# 随机森林的包
import sklearn as skl
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
# print('train_data',train_data)



test_df = pd.read_csv('test.csv')
test_data = test_df.values
print(test_data)
out = pd.read_csv('out-data.csv')
out_data = out.values
print(out_data)




# 2/3的train_data作为训练数据，1/3的train_data作为测试数据来训练模型
num_features = train_data.shape[0] # 拿到train_data的行数，也就是所有数据的个数作为特征值
print("Number of all features: \t\t", num_features)
split = int(num_features * 2/3) # 这里是取2/3行作为训练 后1/3行作为测试


train = train_data[:split] # 取出前2/3行作为训练数据

# print("train[:,1:]",train[:,1:])


test = train_data[split:] # 取出后1/3行作为测试数据


correctAnswer = pd.read_csv('correctAnswer.csv')





num_features = train_data.shape[0] # 拿到train_data的行数，也就是所有数据的个数作为特征值
print("Number of all features: \t\t", num_features)
split = int(num_features * 2/3) # 这里是取2/3行作为训练 后1/3行作为测试
test = train_data[split:] # 取出后1/3行作为测试数据
# 利用



# clf = RandomForestClassifier(n_estimators=100) # 100 trees
# clf = RandomForestRegressor(n_estimators=100)
# clf = DecisionTreeRegressor()
clf = DecisionTreeClassifier()




# 用全部训练数据来做训练
target = train_data[:,0].ravel()
train = train_data[:,1:]
model = clf.fit(train, target)




# 利用append添加数据




# 用测试集数据来预测最终结果
output = model.predict(test_data).T
print(model.predict(test_data))
print('output',output)

resm = np.sqrt(((output - out_data) ** 2).mean())

# print(correctAnswer)

# 计算均方根误差（RMSE）

# for i in output:
#     for j in correctAnswer:
#         a = (i - j) ** 2
#
#         rmse = sp.sqrt(sp.mean()/30)
#         print(rmse)


# def rmse(y_test, y):
#     return sp.sqrt(sp.mean((y_test - y) ** 2))







# 输出预测结果
pd.DataFrame(output).to_csv('out-data.csv', index=False, header=True)