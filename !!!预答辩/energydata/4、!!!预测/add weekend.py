from csv import reader
import numpy as np
import pandas as pd





def load_csv(filename):  #导入csv文件
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

X= load_csv('data.csv')
# print('----------data.csv导入成功----------')


# minutes = [] #早就实现了
# 增加是否是周末
print('数据集共'+str(len(X)/144)+'天') #136天
# print(X[576][0])
#只能用数据集行数来判断

#如何判断周末呢
#0-144都是0
#145-288都是0
#289-432都是0
#433-576都是0
#577-720都是1
#721-864都是1

# 共19个周末

i = 144*4
for a in range(19):
    for j in range(144*2):
        b=i+j
        X[b][26]=1
        print('第'+str(b+1)+'行的周末数据为'+ str(X[b][26]))
    i = i + 144*7

# if(周末)：
#   X[][26]=1

# 2016/1/12周二
# for i in range(len(X)):
#     if(len(X)/144)