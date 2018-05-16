from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor



# tree_model 训练的模型
# indoorTrainData_prepared 特征
# indoorTrainData_labels 目标

train_df = pd.read_csv('train.csv')
# 将DataFrame的数据转换成Array
train_data = train_df.values

train = train_data[:,1:]
target = train_data[:,0].ravel()


test_df = pd.read_csv('test.csv')
test_data = test_df.values
print(test_data)
out = pd.read_csv('out.csv')
out_data = out.values
print(out_data)


clf = RandomForestRegressor(n_estimators=100)
model = clf.fit(train, target)

# scores = cross_val_score(tree_model,indoorTrainData_prepared,indoorTrainData_labels,scoring='neg_mean_squared_error',cv=10)
scores = cross_val_score(model,test_data,out_data,scoring='mean_squared_error',cv=10)

tree_scores = np.sqrt(-scores)

print('tree_model')
print(tree_scores)
print(tree_scores.mean())
print(tree_scores.std())