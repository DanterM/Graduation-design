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



out = pd.read_csv('out_1.csv')

correctAnswer = pd.read_csv('correctAnswer.csv')


out00 = pd.read_csv('out_00.csv')
out11 = pd.read_csv('out_11.csv')
# for i in out:
#     print(i)
#
# for j in correctAnswer:
#     print(j)



# for i,j in zip(out,correctAnswer):
#     print(out - correctAnswer)

a = [60,60,60,60,60,60,60,60,60,60,60,60,60,110,90,110,110,110,110,110,110,110,110,110,110,110,110,110,110,110]
b = [60,60,50,50,60,50,60,60,60,70,230,580,430,250,100,100,90,70,80,140,120,190,110,110,110,110,100,100,100,100]



# 计算均方误差RMSE
d = 0
for i in range(30):
    c = a[i]-b[i]
    d += (c**2)
print(math.sqrt(d/30))

# 计算均方误差
d = 0
sum = 0
for i in b:
    sum += i
mean = sum / 30
print("平均值为",mean)


up = 0
down = 0
for i in range(30):
    c = a[i]-b[i]
    up += c**2
    down += (a[i] - mean)**2
print("R方值为",1-(up/down))


# d = 0
# for i in range(30):
#     c = a[i]-b[i]
#     d += (c**2)/30
#     print(d)
# print('math.sqrt(d)',math.sqrt(d))


# for x,y in zip(a,b):
#     print(a - b)


# a = out-correctAnswer
# print(a)

# resm = np.(((out - correctAnswer) ** 2).mean())
# print(resm)