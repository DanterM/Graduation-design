from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from IPython.display import Image
from sklearn import tree
import pydotplus
from csv import reader



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
print(names)


# 训练模型，限制树的最大深度4
# clf = RandomForestClassifier(max_depth=5)


# RandomForestRegressor()参数
# max_features 随机森林允许单个决策树使用特征的最大数量 增加max_features一般能提高模型的性能，因为在每个节点上，我们有更多的选择可以考虑。 然而，这未必完全是对的，因为它降低了单个树的多样性，而这正是随机森林独特的优点。 但是，可以肯定，你通过增加max_features会降低算法的速度。 因此，你需要适当的平衡和选择最佳max_features。
# n_estimators 建立子树的数量 较多的子树可以让模型有更好的性能，但同时让你的代码变慢。 你应该选择尽可能高的值，只要你的处理器能够承受的住，因为这使你的预测更好更稳定。
# min_sample_leaf 最小样本叶片大小
# n_jobs 这个参数告诉引擎有多少处理器是它可以使用。 “-1”意味着没有限制，而“1”值意味着它只能使用一个处理器
# random_state 此参数让结果容易复现。 一个确定的随机值将会产生相同的结果，在参数和训练数据不变的情况下。 我曾亲自尝试过将不同的随机状态的最优参数模型集成，有时候这种方法比单独的随机状态更好。
# random_state 是随机数生成器使用的种子; 如果是RandomState实例，random_state就是随机数生成器; 如果为None，则随机数生成器是np.random使用的RandomState实例。
# oob_score 这是一个随机森林交叉验证方法。 它和留一验证方法非常相似，但这快很多。 这种方法只是简单的标记在每颗子树中用的观察数据。 然后对每一个观察样本找出一个最大投票得分，是由那些没有使用该观察样本进行训练的子树投票得到。

# criterion string, optional (default=”gini”) 字符串，可选择(默认值为“gini”)。 衡量分裂质量的性能（函数）。
# max_depth （决策）树的最大深度
# min_samples_split 分割内部节点所需要的最小样本数量
# min_samples_leaf 需要在叶子结点上的最小样本数量
# max_leaf_nodes 以最优的方法使用max_leaf_nodes来生长树
# bootstrap 建立决策树时，是否使用有放回抽样。
# estimators_  决策树分类器的序列
# feature_importances_ 特征的重要性（值越高，特征越重要）
# oob_score_  使用袋外估计获得的训练数据集的得分。



# clf = RandomForestRegressor(max_depth=6,min_sample_leaf=100)
clf = RandomForestRegressor(max_depth=6,oob_score='true',n_estimators=10)

#拟合模型
clf.fit(X, y)


import numpy as np
import matplotlib.pyplot as plt
y_importances = clf.feature_importances_
data = 'feature_names-26.csv'
x_importances = load_csv(data)[0]
y_pos = np.arange(len(x_importances))
plt.barh(y_pos, y_importances, align='center')
plt.yticks(y_pos, x_importances)
plt.xlabel('Importances')
plt.xlim(0,0.5)
plt.title('Features Importances')
plt.tight_layout()
plt.show()




# 多种训练模型 也就是每一个决策树
Estimators = clf.estimators_

# print(type(Estimators))  #//<class 'list'>

# print('Estimators',Estimators)


for index, model in enumerate(Estimators):
    filename = 'energydata_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model , out_file=None,
                         feature_names=names,
                         class_names=names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    Image(graph.create_png())
    graph.write_pdf(filename)

print('----------已生成文件----------')