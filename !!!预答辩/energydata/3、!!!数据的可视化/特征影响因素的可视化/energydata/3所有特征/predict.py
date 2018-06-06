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







# 选择训练模型
# clf = DecisionTreeClassifier(max_depth=4) # date相关性最高
# clf = DecisionTreeRegressor(max_depth=4) # date相对最高
# clf = RandomForestClassifier(oob_score = 'true',random_state =50) # 平均
clf = RandomForestRegressor(oob_score = 'true',random_state =50) # 平均



#拟合模型
clf.fit(X, y)
predict = clf.predict(X)
print(predict)
pd.DataFrame({"Id": range(1, len(predict)+1), "Label": predict}).to_csv('predict.csv', index=False, header=True)

# predicted_data = load_csv('predict.csv')
# print('----------predict.csv数据导入成功----------')

print('----------已处理predict.csv数据----------')

# print(predicted_data[0][0])







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

