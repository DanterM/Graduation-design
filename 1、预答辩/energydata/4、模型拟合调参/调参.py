# 加上时间特征之后数据的格式发生变化 Appliances不是第一列了 ？？？可能会产生问题 已解决

import numpy as np
import pandas as pd
from csv import reader
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import matplotlib
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




# print(len(train_data))# 19584
# print(train_data)
# print(len(train_data))
# a = train_data[:2]
# print(a)
# print(int(len(train_data)*2/3))
# print(type(X[0][0]))
#
#
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

timelist = []
acclist = []

# range(1,1000,100)

for time in range(1,100):
    clf = RandomForestRegressor(n_estimators=100, oob_score='true', max_depth=2,max_features='log2',min_samples_leaf=time,min_samples_split=time)
    # min_samples_split min_samples_leaf 浮动很大 不考虑该参数
    # clf = KNeighborsRegressor(n_neighbors=time)



    clf.fit(X, y)
    predict_data = train_data[int(len(train_data) * 9 / 10) + 1:]
    predict_target = clf.predict(predict_data)
    correct_target = train_target[int(len(train_target) * 9 / 10) + 1:]

    a = 0
    wucha = 50
    for i in range(len(predict_data)):
        if abs((predict_target - correct_target)[i]) < wucha:
            a += 1
    acc = a / len(predict_data)
    # print('max_depth='+str(time))
    # print('误差在' + str(wucha) + 'Wh内正确率：', acc, '\n')

    timelist.append(time)
    acclist.append(acc)
    plt.plot(timelist, acclist)

matplotlib.rcParams['font.family']='SimHei'
my_x_ticks = np.arange(0, 100, 10)
plt.xticks(my_x_ticks)  # 由于横轴的数据太长，旋转90度，竖着显示
plt.xticks(fontsize=10)
plt.xlabel("canshu", fontproperties='SimHei')  # 指定横轴和纵轴的标签
plt.ylabel("acc")
# plt.title("最大深度-正确率")  # 标题
plt.show()

    # RMSE
    # rmse = np.sqrt(((predict_target - correct_target) ** 2).mean())

    # R方
    # average = np.sum(correct_target) / len(correct_target)  # 平均值
    # a = []
    # for i in range(len(correct_target)):
    #     a.append(average)
    # r = 1 - (((predict_target - correct_target) ** 2).sum() / (((predict_target - a) ** 2).sum()))

    # MAE
    # mae = (abs((predict_target - correct_target)).sum()) / len(correct_target)

    # MAPE
    # mape = (abs((predict_target - correct_target)) / predict_target).sum() / len(correct_target)
