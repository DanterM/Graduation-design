from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from IPython.display import Image
from sklearn import tree
import pydotplus
from csv import reader


# filename = 'data-200.csv'


# 仍然使用自带的iris数据
iris = datasets.load_iris()

def load_csv(filename):  #导入csv文件
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# dataset = load_csv(filename)


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())



# str_column_to_float(dataset)

data = 'data-200.csv'
X= load_csv(data)
for i in range(len(X[0])):
    str_column_to_float(X, i)


# print(X)


target = 'target-200.csv'
y = load_csv(target)[0]
# print(len(y))
# for i in range(len(y)):
#     str_column_to_float(y, i)
# print(y)


feature = 'feature_names-10.csv'
names = load_csv(feature)[0]



# X = iris.data
# print(X)
# y = iris.target
# print(y)

# print(iris.feature_names)
# print(iris.target_names)

# 训练模型，限制树的最大深度4
clf = RandomForestClassifier(max_depth=5)
#拟合模型
clf.fit(X, y)

Estimators = clf.estimators_
for index, model in enumerate(Estimators):
    filename = 'iris_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model , out_file=None,
                         feature_names=names,
                         # class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # 使用ipython的终端jupyter notebook显示。
    Image(graph.create_png())
    graph.write_pdf(filename)