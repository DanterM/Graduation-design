# 加上时间特征之后数据的格式发生变化 Appliances不是第一列了 ？？？可能会产生问题 已解决

import numpy as np
import pandas as pd


# values = pd.read_csv('1-1、数据集energydata-del1day.csv')
# values = values[:200]
#
# print(len(values))
#
# for i in range(len(values)):
#     chuli = values['date'][i].split(':')
#     values['date'][i] = int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1
#     # print('第',i,'行数据',values['date'][i])
# time_manage = values
#
#
# print(time_manage.values)
#
# X = time_manage.values
# print(X)



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


print('----------data.csv时间格式已更改-----------')

X = train_data[:int(len(train_data)*9/10)]
# print(len(X))# 13056
print('----------data.csv导入成功----------')




train_target = load_csv('target.csv')
print('----------target.csv导入成功----------')
# 处理结果target数组 得到y
a = []
for i in train_target:
    # print(i)
    a.append(float(i[0]))
    # a.append(i[0])

train_target = a
y = train_target[:int(len(train_target)*9/10)]
# print(len(y))# 13056
print('----------target.csv处理完成----------')

# print(y)




# 选择训练模型
# clf = DecisionTreeClassifier(max_depth=4) # date相关性最高 //0.14
# clf = DecisionTreeRegressor(max_depth=10) # date相对最高 //0.16
# clf = RandomForestClassifier(oob_score = 'true',random_state =50) # 平均 //0.14
clf = RandomForestRegressor(n_estimators=100,oob_score = 'true') # 平均 // 0.17
# clf = RandomForestRegressor(n_estimators=1000,oob_score = 'true') # 平均 //



#拟合模型
clf.fit(X, y)

# 预测使用特征
predict_data = train_data[int(len(train_data)*9/10)+1:]
# print(predict_data)

predict_target = clf.predict(predict_data)
print('----------预测数据----------')
print(predict_target)
pd.DataFrame({"Id": range(1, len(predict_target)+1), "Label": predict_target}).to_csv('predict_target.csv', index=False, header=True)

# predicted_data = load_csv('predict.csv')
print('----------predict.csv数据导出成功----------','\n')

# print('----------已处理predict.csv数据----------')

# print(predicted_data[0][0])


correct_target = train_target[int(len(train_target)*9/10)+1:]
print(correct_target)
pd.DataFrame({"Id": range(1, len(correct_target)+1), "Label": correct_target}).to_csv('correct_target.csv', index=False, header=True)
# print(len(train_target)) # 19584
# print(len(train_data)) # 19584
# print(len(predict_target)) # 6528
# print(train_data[13056])
# print(predict_data[0])
# print(len(correct_target)) # 6528

# 误差在10Wh内正确率
a = 0
wucha = 1000
for i in range(len(predict_data)):
    if abs((predict_target - correct_target)[i])<wucha:
        a += 1
acc = a/len(predict_data)



# print('误差在5Wh内正确率：',acc,'\n')  #// 0.583078022875817
print('误差在10Wh内正确率：',acc,'\n')  #// 0.7617442810457516
# print('误差在20Wh内正确率：',acc,'\n')  #// 0.8401756535947712



# 评价函数

# 计算RMSE
print('----------RMSE----------')
rmse = np.sqrt(((predict_target - correct_target) ** 2).mean())
print(rmse,'\n')




# R方 越接近1 越好
average = np.sum(correct_target)/len(correct_target) # 平均值

a = []
for i in range(len(correct_target)):
    a.append(average)
r = 1 - (((predict_target - correct_target)**2).sum()/(((predict_target - a)**2).sum()))
print('----------R方----------')
print(r,'\n')


# MAE 越小越好
mae = (abs((predict_target - correct_target)).sum())/len(correct_target)
print('----------平均绝对误差（MAE）----------')
print(mae,'\n')

# MAPE (0-1)之间 越小越好
mape = (abs((predict_target - correct_target))/predict_target).sum()/len(correct_target)
print('----------平均绝对百分误差（MAPE）----------')
print(mape,'\n')
