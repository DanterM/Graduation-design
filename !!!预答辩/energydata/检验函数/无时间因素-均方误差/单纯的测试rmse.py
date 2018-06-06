#30个数据集的验证


# 引入需要的包 数据处理的常用包

import numpy as np
import pandas as pd
import scipy as sp
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
import math



# out = pd.read_csv('out_1.csv')
#
# correctAnswer = pd.read_csv('correctAnswer.csv')
#
#
# out00 = pd.read_csv('out_00.csv')
# out11 = pd.read_csv('out_11.csv')
# for i in out:
#     print(i)
#
# for j in correctAnswer:
#     print(j)



# for i,j in zip(out,correctAnswer):
#     print(out - correctAnswer)

# a = [60,60,60,60,60,60,60,60,60,60,60,60,60,110,90,110,110,110,110,110,110,110,110,110,110,110,110,110,110,110]
correctAnswer = [60,60,50,50,60,50,60,60,60,70,230,580,430,250,100,100,90,70,80,140,120,190,110,110,110,110,100,100,100,100]

RandomForestClassifier = [ 40,60,60,60,60,60,60,60,60,60,60,60,60,60,40,60,60,60,60,110,110,110,110,110,110,110,110,110,110,110]
RandomForestRegressor = [263,275.6,255.2,258,263.2,268.8,250.4 ,273.6 ,267.2 ,265.8 ,256.2, 258.6,253.2, 259.6, 228.4, 233.8, 226,  271.5, 267.1 ,270.8 ,268.7, 277.9, 265.7, 271.2,265.5 ,261.9, 262.4 ,251.2 ,243.1 ,238 ]
DecisionTreeRegressor = [620,620,620,620,620,620,620,620,620,620,60,60,60,60,260,260,260,90,90,90,260,90,90,90,140,110,140,110,110,110]
DecisionTreeClassifier = [380., 380., 380., 380., 380., 380., 380., 380., 380., 380., 380., 360., 380., 360.,
  80.  ,80., 100. ,100., 100., 410. , 90. ,410., 110., 110., 110., 110. ,110. ,110.,
 110. ,110.]


a = RandomForestClassifier


# 计算均方误差RMSE
d = 0
for i in range(30):
    c = a[i]-correctAnswer[i]
    d += (c**2)
print('均方误差为',math.sqrt(d/30))


d = 0
sum = 0
for i in correctAnswer:
    sum += i
mean = sum / 30
print("平均值为",mean)


up = 0
down = 0
for i in range(30):
    c = a[i]-correctAnswer[i]
    up += c**2
    down += (a[i] - mean)**2
print("R方值为",1-(up/down))




# DecisionTreeClassifier()
# 均方误差为 203.91991892243715
# 平均值为 126.66666666666667
# R方值为 -0.18851027342881665


# DecisionTreeRegressor()
# 均方误差为 353.4308041658697
# 平均值为 126.66666666666667
# R方值为 -0.47989890214042963



# RandomForestRegressor()
# 均方误差为 175.3460844539545
# 平均值为 126.66666666666667
# R方值为 -0.7366396861293707


# # RandomForestClassifier()
# 均方误差为 127.47548783981962
# 平均值为 126.66666666666667
# R方值为 -4.206479174083301
