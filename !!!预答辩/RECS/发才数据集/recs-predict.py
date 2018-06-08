# 加上时间特征之后数据的格式发生变化 Appliances不是第一列了 ？？？可能会产生问题 已解决

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

print('----------data.csv处理完成-时间格式已更改----------')



# 选择训练模型
# clf = DecisionTreeClassifier(max_depth=4) # date相关性最高 //0.2519403594771242
# clf = DecisionTreeRegressor(max_depth=4) # date相对最高 //0.3088235294117647
# clf = RandomForestClassifier(oob_score = 'true',random_state =50) # 平均 //0.9693116830065359



# clf = RandomForestRegressor(n_estimators=10,oob_score = 'true') # 平均 // 0.7604166666666666
# clf = RandomForestRegressor(n_estimators=10,max_depth=1000) # 平均 // 0.7604166666666666


clf = RandomForestRegressor(n_estimators=100,oob_score = 'true') # 平均 //0.7679738562091504
# clf = KNeighborsRegressor(n_neighbors=2)   #0.3472732843137255
# clf = KNeighborsRegressor(n_neighbors=10)   #0.3217933006535948
# clf = KNeighborsRegressor(n_neighbors=100)   #0.2907986111111111
# clf = SVC()



#拟合模型
clf.fit(X, y)


predict = clf.predict(X)
print('----------预测数据----------')
print(predict)
pd.DataFrame({"Id": range(1, len(predict)+1), "Label": predict}).to_csv('predict.csv', index=False, header=True)

# predicted_data = load_csv('predict.csv')
print('----------predict.csv数据导出成功----------','\n')

# print('----------已处理predict.csv数据----------')

# print(predicted_data[0][0])




# 误差在10Wh内正确率
a = 0
wucha = 1000
for i in range(len(y)):
    if abs((predict - y)[i])<wucha:
        a += 1
acc = a/len(y)



# print('误差在5Wh内正确率：',acc,'\n')  #// 0.583078022875817
print('误差在1000内正确率：',acc,'\n')  #// 0.7617442810457516
# print('误差在20Wh内正确率：',acc,'\n')  #// 0.8401756535947712



# 评价函数

# 计算RMSE
print('----------RMSE----------')
rmse = np.sqrt(((predict - y) ** 2).mean())
print(rmse,'\n')




# R方 越接近1 越好
average = np.sum(y)/len(y) # 平均值
a = []
for i in range(len(y)):
    a.append(average)
r = 1 - (((predict - y)**2).sum()/(((predict - a)**2).sum()))
print('----------R方----------')
print(r,'\n')


# MAE 越小越好
mae = (abs((predict - y)).sum())/len(y)
print('----------平均绝对误差（MAE）----------')
print(mae,'\n')

# MAPE (0-1)之间 越小越好
mape = (abs((predict - y))/predict).sum()/len(y)
print('----------平均绝对百分误差（MAPE）----------')
print(mape,'\n')


















