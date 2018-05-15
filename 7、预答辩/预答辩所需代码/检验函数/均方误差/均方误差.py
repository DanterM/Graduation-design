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





num_features = train_data.shape[0] # 拿到train_data的行数，也就是所有数据的个数作为特征值
print("Number of all features: \t\t", num_features)
split = int(num_features * 2/3) # 这里是取2/3行作为训练 后1/3行作为测试

# 利用
clf = RandomForestClassifier(n_estimators=100) # 100 trees

# 用全部训练数据来做训练
target = train_data[:,0].ravel()
train = train_data[:,1:]
model = clf.fit(train, target)




# 用测试集数据来预测最终结果
output = model.predict(test_data)
for i in output:
    print(i)

print(output)


for j in correctAnswer:
    print(j)




print(correctAnswer)


# 输出预测结果
pd.DataFrame(output).to_csv('out0.csv', index=False, header=True)