# from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from csv import reader
from IPython.display import Image
from sklearn import tree
import pydotplus

# 仍然使用自带的iris数据

# iris = datasets.load_iris()
# X = iris.data
# print(X)
# y = iris.target
# print(y)



def load_csv(filename):  #导入csv文件
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# def column_to_float(dataSet):
#     featLen=len(dataSet[0])
#     for data in dataSet:
#         for column in range(featLen):
#             data[column]=float(data[column].strip())


X= load_csv('data-200.csv')
print(X)

y = load_csv('target-200.csv')[0]
print(y)
feature = 'feature_names-10.csv'
names = load_csv(feature)[0]


# 训练模型，限制树的最大深度4
# clf = DecisionTreeClassifier(max_depth=4)
clf = RandomForestRegressor(n_estimators=100)
#拟合模型
clf.fit(X, y)

y_importances = clf.feature_importances_



data = 'feature_names-10.csv'
x_importances = load_csv(data)[0]
# print(target)


# x_importances = iris.feature_names

print(x_importances)

y_pos = np.arange(len(x_importances))
print(y_pos)
print(x_importances)

# # 横向柱状图
# plt.barh(y_pos, y_importances, align='center')
# plt.yticks(y_pos, x_importances)
# plt.xlabel('Importances')
# plt.xlim(0,1)
# plt.title('Features Importances')
# plt.tight_layout()
# plt.show()


# 竖向柱状图
plt.bar(y_pos, y_importances, width=0.4, align='center', alpha=0.4)
plt.xticks(y_pos, x_importances)
plt.ylabel('Importances')
plt.ylim(0,1)
plt.title('Features Importances')
plt.show()




# plt.title('特征重要性')
# plt.bar(y_pos, y_importances)
# plt.xticks(y_pos, x_importances)
# plt.xlim([-1,1])
# plt.tight_layout()
# plt.show()