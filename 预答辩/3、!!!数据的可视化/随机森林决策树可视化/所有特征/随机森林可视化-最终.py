from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
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


data = 'data.csv'
X= load_csv(data)
# for i in range(len(X[0])):
#     str_column_to_float(X, i)
for i in range(len(X)):
    chuli = X[i][0].split(':')
    X[i][0] = int(chuli[0]) * 60 + (int(chuli[1]))

# print(X)


target = 'target.csv'
y = load_csv(target)[0]
# print(len(y))
# for i in range(len(y)):
#     str_column_to_float(y, i)
# print(y)

y = load_csv('target.csv')
# 处理结果target数组 得到y
a = []
for i in y:
    # print(i)
    a.append(float(i[0]))
    # a.append(i[0])

y = a


feature = 'feature_names-26.csv'
names = load_csv(feature)[0]



# 训练模型，限制树的最大深度4
# clf = RandomForestClassifier(max_depth=5)
clf = RandomForestRegressor(max_depth=5)
#拟合模型
clf.fit(X, y)



# 多种训练模型 也就是每一个决策树
Estimators = clf.estimators_

# print(type(Estimators))  #//<class 'list'>

print('Estimators',Estimators)


for index, model in enumerate(Estimators):
    filename = 'energydata_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model , out_file=None,
                         feature_names=names,
                         # class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    Image(graph.create_png())
    graph.write_pdf(filename)
