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
print(iris)

X = iris.data
y = iris.target

# 训练模型，限制树的最大深度4
clf = DecisionTreeClassifier(max_depth=4)
#拟合模型
clf.fit(X, y)

dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
# 使用ipython的终端jupyter notebook显示。
Image(graph.create_png())
# 如果没有ipython的jupyter notebook，可以把此图写到pdf文件里，在pdf文件里查看。
graph.write_pdf("iris.pdf")