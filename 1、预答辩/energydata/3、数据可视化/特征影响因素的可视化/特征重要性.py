# 加上时间特征之后数据的格式发生变化 Appliances不是第一列了 ？？？可能会产生问题 已解决

import numpy as np
import pandas as pd
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

# 全部数据集导入
X= load_csv('data.csv')
# data1中末尾多留了一行 为了加上周末因素

# print(len(X))
# print(type(X[0][0]))
print('----------data.csv导入成功----------')


y_data = load_csv('target.csv')
print('----------target.csv导入成功----------')
# 处理结果target数组 得到y
#定义训练输出数组
a = []
for i in y_data:
    # print(i)
    a.append(float(i[0]))
    # a.append(i[0])

y = a
print('----------target.csv处理完成----------')

# print(y)

# 增加是否是周末特征
i = 144*4
for a in range(19):
    for j in range(144*2):
        b=i+j
        X[b][26]=1
        # print('第'+str(b+1)+'行的周末数据为'+ str(X[b][26]))
    i = i + 144*7


# 四种时间处理方式

# 1、1-144处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1


# 2、秒数处理时间
# 从日期/时间变量可以生成其他额外的特性:每天从午夜开始的秒数(NSM)、周状态(周末或工作日)和一周的天数。


# for i in range(len(X)):
#     # print(X[i][0])
#     chuli = X[i][0].split(':')
#     X[i][0] = int(chuli[0]) * 3600 + (int(chuli[1])*60)
# print('----------data.csv处理完成-时间格式已更改----------')

# print(X)


# 3、分钟数处理时间
for i in range(len(X)):
    chuli = X[i][0].split(':')
    X[i][0] = int(chuli[0]) * 60 + (int(chuli[1]))


# 4、0-1分布处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = (int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1)/len(X)




# 数据预处理结束
print("--数据预处理结束--")





# 训练模型，限制树的最大深度X
# clf = DecisionTreeClassifier(max_depth=4) # date相关性最高
# clf = DecisionTreeRegressor(max_depth=10) # date相对最高
# clf = RandomForestClassifier(oob_score = 'true',random_state =50) # 平均
clf = RandomForestRegressor(oob_score = 'true',random_state =50) # 平均
#拟合模型
clf.fit(X, y)
y_importances = clf.feature_importances_
print('----------y_importances----------','\n',y_importances)

# feature = 'feature_names-26.csv'
# names = load_csv(feature)[0]

data = 'feature_names-26.csv'
x_importances = load_csv(data)[0]

print('----------x_importances----------','\n',x_importances)

y_pos = np.arange(len(x_importances))

print('----------y_pos----------','\n',y_pos)

plt.barh(y_pos, y_importances, align='center')
plt.yticks(y_pos, x_importances)
plt.xlabel('Importances')
plt.xlim(0,0.3)
plt.title('Features Importances')
plt.tight_layout()
plt.show()