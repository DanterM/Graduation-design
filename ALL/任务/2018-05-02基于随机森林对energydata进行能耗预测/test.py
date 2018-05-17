import numpy as np
import pandas as pd
from sklearn import tree
# 随机森林的包
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
import pydotplus
from IPython.display import Image
# 画图的包
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

# 读取数据（请先去 https://www.kaggle.com/c/digit-recognizer/data 上下载数据）
# 读取成DataFrame的数据
train_df = pd.read_csv('train.csv')
# 将DataFrame的数据转换成Array
train_data = train_df.values

test_df = pd.read_csv('test.csv')
test_data = test_df.values

correctAnswer = pd.read_csv('correctAnswer.csv')

print(train_df.head())
print(train_data)
print(correctAnswer)