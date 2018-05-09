# from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn import tree
import pydotplus

# 仍然使用自带的iris数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 训练模型，限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=4)
#拟合模型
clf.fit(X, y)

y_importances = clf.feature_importances_
x_importances = iris.feature_names
y_pos = np.arange(len(x_importances))
# 横向柱状图
plt.barh(y_pos, y_importances, align='center')
plt.yticks(y_pos, x_importances)
plt.xlabel('Importances')
plt.xlim(0,1)
plt.title('Features Importances')
plt.show()

# 竖向柱状图
plt.bar(y_pos, y_importances, width=0.4, align='center', alpha=0.4)
plt.xticks(y_pos, x_importances)
plt.ylabel('Importances')
plt.ylim(0,1)
plt.title('Features Importances')
plt.show()