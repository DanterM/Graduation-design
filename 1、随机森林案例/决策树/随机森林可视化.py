from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
from sklearn import tree
import pydotplus

# 仍然使用自带的iris数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 训练模型，限制树的最大深度4
clf = RandomForestClassifier(max_depth=4)
#拟合模型
clf.fit(X, y)

Estimators = clf.estimators_
for index, model in enumerate(Estimators):
    filename = 'iris_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model , out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # 使用ipython的终端jupyter notebook显示。
    Image(graph.create_png())
    graph.write_pdf(filename)