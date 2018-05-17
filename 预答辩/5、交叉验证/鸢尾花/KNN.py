from sklearn.datasets import load_iris  # iris数据集
from sklearn.model_selection import train_test_split  # 分割数据模块
from sklearn.neighbors import KNeighborsClassifier  # K最近邻(kNN，k-NearestNeighbor)分类算法
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# 加载iris数据集
iris = load_iris()
X = iris.data
# print(X)
# [[5.1 3.5 1.4 0.2]
#  [4.9 3.  1.4 0.2]
#  [4.7 3.2 1.3 0.2]
#  [4.6 3.1 1.5 0.2]
#  [5.  3.6 1.4 0.2]
#  [5.4 3.9 1.7 0.4]
#  [4.6 3.4 1.4 0.3]
#  [5.  3.4 1.5 0.2]
#  [4.4 2.9 1.4 0.2]
#  [4.9 3.1 1.5 0.1]
#  [5.4 3.7 1.5 0.2]
#  [4.8 3.4 1.6 0.2]
#  [4.8 3.  1.4 0.1]]
y = iris.target
# print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2]

# print(iris.target_names)
# 分割数据并
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)


# 建立模型
knn = KNeighborsClassifier()
# randomForest = RandomForestClassifier(n_estimators=30)
# 训练模型
knn.fit(X_train, y_train)
# randomForest.fit(X_train,y_train)

# 将准确率打印出
print(knn.score(X_test, y_test))
# print(randomForest.score(X_test, y_test))
# 0.973684210526     基础验证的准确率





from sklearn.cross_validation import cross_val_score  # K折交叉验证模块

# 使用K折交叉验证模块
scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
# scores = cross_val_score(randomForest, X, y, cv=5, scoring='accuracy')

KFold(4, n_folds=2)


# 将5次的预测准确率打印出
print(scores)
# [ 0.96666667  1.          0.93333333  0.96666667  1.        ]

# 将5次的预测准确平均率打印出
print(scores.mean())
# 0.973333333333



import matplotlib.pyplot as plt  # 可视化模块

# 建立测试参数集
k_range = range(1, 31)

k_scores = []

# 藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # random = RandomForestRegressor()

    scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
    # scores = cross_val_score(random, X, y, cv=10, scoring='accuracy')

    k_scores.append((scores).mean())

# 可视化数据
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

