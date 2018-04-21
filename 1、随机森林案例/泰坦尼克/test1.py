import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
# import matplotlib.pylab as plt


# 导入数据，顺便看看数据的类别分布
train = pd.read_csv('/Users/Jarvis/PycharmProjects/titanic/train_modified.csv')
target = 'Disbursed'  # Disbursed的值就是二元分类的输出
IDcol = 'ID'
train['Disbursed'].value_counts()

# 可以看到类别输出如下，也就是类别0的占大多数：

# 接着选择好样本特征和类别输出，样本特征为除去ID和输出类别的列
x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']

# 不管任何参数，都用默认的，拟合下数据看看
rf0 = RandomForestClassifier(oob_score=True, random_state=10)
rf0.fit(X, y)
print(rf0.oob_score_)
y_predprob = rf0.predict_proba(X)[:, 1]
print("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))
# 输出如下：0.98005  AUC Score (Train): 0.999833
# 可见袋外分数已经很高（理解为袋外数据作为验证集时的准确率，也就是模型的泛化能力），而且AUC分数也很高（AUC是指从一堆样本中随机抽一个，抽到正样本的概率比抽到负样本的概率 大的可能性）。相对于GBDT的默认参数输出，RF的默认参数拟合效果对本例要好一些。

# 首先对n_estimators进行网格搜索
param_test1 = {'n_estimators': range(10, 71, 10)}
gsearch1 = GridSearchCV(estimator=RandomForestClassifier(min_samples_split=100,min_samples_leaf=20, max_depth=8, max_features='sqrt',random_state=10),param_grid=param_test1, scoring='roc_auc', cv=5)
gsearch1.fit(X, y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_


# 这样我们得到了最佳的弱学习器迭代次数，接着我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(50, 201, 20)}
gsearch2 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=60,min_samples_leaf=20, max_features='sqrt', oob_score=True,random_state=10),param_grid=param_test2, scoring='roc_auc', iid=False, cv=5)
gsearch2.fit(X, y)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_

# 已经取了三个最优参数，看看现在模型的袋外分数：
rf1 = RandomForestClassifier(n_estimators=60, max_depth=13, min_samples_split=110,min_samples_leaf=20, max_features='sqrt', oob_score=True, random_state=10)
rf1.fit(X, y)
print(rf1.oob_score_)
# 输出结果为：0.984
# 可见此时我们的袋外分数有一定的提高。也就是时候模型的泛化能力增强了。对于内部节点再划分所需最小样本数min_samples_split，我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。下面我们再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。

# 再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
param_test3 = {'min_samples_split': range(80, 150, 20), 'min_samples_leaf': range(10, 60, 10)}
gsearch3 = GridSearchCV(estimator=RandomForestClassifier(n_estimators=60, max_depth=13,max_features='sqrt', oob_score=True, random_state=10),param_grid=param_test3, scoring='roc_auc', iid=False, cv=5)
gsearch3.fit(X, y)
gsearch3.grid_scores_, gsearch2.best_params_, gsearch2.best_score_



param_test4= {'max_features':range(3,11,2)}
gsearch4= GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60,max_depth=13, min_samples_split=120,min_samples_leaf=20 ,oob_score=True, random_state=10),param_grid = param_test4,scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X,y)
gsearch4.grid_scores_,gsearch4.best_params_, gsearch4.best_score_




#用我们搜索到的最佳参数，我们再看看最终的模型拟合：
rf2= RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)
rf2.fit(X,y)
print(rf2.oob_score_)
#此时的输出为：0.984
#可见此时模型的袋外分数基本没有提高，主要原因是0.984已经是一个很高的袋外分数了，如果想进一步需要提高模型的泛化能力，我们需要更多的数据。