# 平均不纯度减少 mean decrease impurity
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np


boston = load_boston()
X = boston['data']
y = boston['target']
names = boston['feature_names']

rf = RandomForestRegressor()
rf.fit(X,y)

print("Feature sorted by their score:")
print(sorted(zip(map(lambda x:round(x,4),rf.feature_importances_),names),reverse=True))

# 需要注意的一点是，关联特征的打分存在不稳定的现象，这不仅仅是随机森林特有的，大多数基于模型的特征选择方法都存在这个问题。