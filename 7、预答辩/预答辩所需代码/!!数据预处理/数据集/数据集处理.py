# 1、处理日期格式
# 2、

import numpy as np
import pandas as pd


values = pd.read_csv('1-energydata-del1day.csv')

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




# 时间的格式化（1-144）或者0-1归一化
# test = values['date'][133]
# test1 = test.split(':')
# float(test1[0])*6 + (float(test1[1])/10)
# print(test.split(':'))


for i in range(len(values)):
    chuli = values['date'][i].split(':')
    values['date'][i] = int(chuli[0]) * 6 + (int(chuli[1]) / 10) + 1

# 时间特征处理后得到的数据集 time_manage
time_manage = values
