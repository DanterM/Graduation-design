import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_df = pd.read_csv('data.csv')

train_date = train_df['date']
train_appliances = train_df['Appliances']

# 数据集共136天
# 循环遍历数据集所有天数状态图
# for i in range(135):
#     plt.plot(train_date[i*144:(i+1)*144], train_appliances[i*144:(i+1)*144])

# 横坐标为每一天   纵坐标为每天能量的总值
# 共136天 143

dayall = 0
a = []
for i in range(135):
    # plt.plot(i+1, sum(train_appliances[i * 144:(i + 1) * 144]))
    dayall = train_appliances[i*144:(i+1)*144]
    # print(sum(dayall))
    a.append(sum(dayall))
# print(a)



for j in range(135):
    plt.plot(range(135), a)


plt.xticks(rotation=90,fontsize=10)  #由于横轴的数据太长，旋转90度，竖着显示
plt.xlabel("Time")      #指定横轴和纵轴的标签
plt.ylabel("Appliances")
plt.title("Appliances-Time") #标题
plt.show()
