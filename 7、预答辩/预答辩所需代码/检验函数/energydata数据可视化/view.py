import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestRegressor



train_df = pd.read_csv('../dataset/1energydata-del1day.csv')

train_date = train_df['date']
train_appliances = train_df['Appliances']

#  前144个-一天的数量
# print(train_date.head(144)) ==print(train_date[0:143])
# print(train_appliances.head(144))

# 第二天
# print(train_date[144:288])



# 前三天情况
# plt.plot(train_date[0:144],train_appliances[0:144])
# plt.plot(train_date[144:288],train_appliances[144:288])
# plt.plot(train_date[288:432],train_appliances[288:432])



# 数据集共136天
# 循环遍历数据集所有天数状态图
for i in range(136):
    plt.plot(train_date[i*144:(i+1)*144], train_appliances[i*144:(i+1)*144])


plt.xticks(rotation=90)  #由于横轴的数据太长，旋转90度，竖着显示
plt.xlabel("Time")      #指定横轴和纵轴的标签
plt.ylabel("Appliances")
plt.title("") #标题
plt.show()

