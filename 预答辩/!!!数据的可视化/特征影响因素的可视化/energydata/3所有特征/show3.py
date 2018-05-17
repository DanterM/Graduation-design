# 加上时间特征之后数据的格式发生变化 Appliances不是第一列了 ？？？可能会产生问题


import numpy as np
import pandas as pd


# values = pd.read_csv('1-energydata-del1day.csv')
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

X= load_csv('data-200.csv')




# print(X[i][0]) #//00:00


# 三种时间处理方式

# 1-144处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1


# 秒数处理时间
# 从日期/时间变量可以生成其他额外的特性:每天从午夜开始的秒数(NSM)、周状态(周末或工作日)和一周的天数。

for i in range(len(X)):
    chuli = X[i][0].split(':')
    X[i][0] = int(chuli[0]) * 3600 + (int(chuli[1])*60)




# 分钟数处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = int(chuli[0]) * 60 + (int(chuli[1]))



# 0-1分布处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = (int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1)/len(X)




y = load_csv('target-200.csv')[0]
print(y)


# 训练模型，限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=4,random_state=10) # date效果最好
# clf = DecisionTreeRegressor(max_depth=4) # date相对很少
# clf = RandomForestClassifier(oob_score = 'true',random_state =50) # 平均
# clf = RandomForestRegressor(oob_score = 'true',random_state =50) # date相对很少



#拟合模型
clf.fit(X, y)

y_importances = clf.feature_importances_


feature = 'feature_names-26.csv'
names = load_csv(feature)[0]

data = 'feature_names-26.csv'
x_importances = load_csv(data)[0]


print(x_importances)


y_pos = np.arange(len(x_importances))
print(y_pos)
print('x_importances',x_importances)

plt.barh(y_pos, y_importances, align='center')
plt.yticks(y_pos, x_importances)
plt.xlabel('Importances')
plt.xlim(0,1)
plt.title('Features Importances')
plt.tight_layout()
plt.show()