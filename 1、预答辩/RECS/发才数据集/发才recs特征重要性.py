# 加上时间特征之后数据的格式发生变化 Appliances不是第一列了 ？？？可能会产生问题 已解决

import numpy as np
import pandas as pd

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


# 全部数据集导入
X= load_csv('data.csv')
print(len(X))
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

from sklearn.svm import SVC
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

data = 'feature.csv'
x_importances = load_csv(data)[0]



print('----------x_importances----------','\n',x_importances)
print(len(x_importances))

y_pos = np.arange(len(x_importances))

print('----------y_pos----------','\n',y_pos)


plt.barh(y_pos, y_importances, align='center')
plt.yticks(y_pos, x_importances)
plt.xlabel('Importances')
plt.xlim(0,1)
plt.title('Features Importances')
plt.tight_layout()
plt.show()
