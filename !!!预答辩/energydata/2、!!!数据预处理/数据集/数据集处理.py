# 1、处理日期格式
# 2、

import numpy as np
import pandas as pd


X = pd.read_csv('data.csv')

# print(len(values)) //19584

# 使用源数据时  利用切片删除元素   例：截取出第43行之后的所有数据
# data:从2016-1-12日开始到2016-5-26日的所有数据
# values = values[42:19626]

# for i in range(19584):
#     date = values['date'][i]
#     date.split(':')
#     print(date)
#
#
# print(type(date))


# 四种时间处理方式

# 时间的格式化（1-144）或者0-1归一化
# test = values['date'][133]
# test1 = test.split(':')
# float(test1[0])*6 + (float(test1[1])/10)
# print(test.split(':'))


# print(values['date'][0]) #//00：00

for i in range(len(X)):
    chuli = X['date'][i].split(':')
    X['date'][i] = int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1




# 秒数处理时间
# 从日期/时间变量可以生成其他额外的特性:每天从午夜开始的秒数(NSM)、周状态(周末或工作日)和一周的天数。


# for i in range(len(X)):
#     # print(X[i][0])
#     chuli = X[i][0].split(':')
#     X[i][0] = int(chuli[0]) * 3600 + (int(chuli[1])*60)

# print(X)


# 分钟数处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = int(chuli[0]) * 60 + (int(chuli[1]))



# 0-1分布处理时间
# for i in range(len(X)):
#     chuli = X[i][0].split(':')
#     X[i][0] = (int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1)/len(X)

# 时间特征处理后得到的数据集 time_manage
time_manage = X
print(time_manage)
