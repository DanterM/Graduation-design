
# 毕业设计-基于随机森林对建筑能耗进行预测

环境配置
  8
macOS High Sierra10.13.5，Python3.6，PyCharm


所用数据集-->energydata、RECS

## 实现过程
### 一、数据预处理

#### 1、energydata

##### 数据集概述：
    实例数量：19735个  特征数量：29个  任务类型：回归  无缺失值

##### 原有特征：

```
date time year-month-day hour:minute:second 
Appliances, energy use in Wh 
lights, energy use of light fixtures in the house in Wh 
T1, Temperature in kitchen area, in Celsius 
RH_1, Humidity in kitchen area, in % 
T2, Temperature in living room area, in Celsius 
RH_2, Humidity in living room area, in % 
T3, Temperature in laundry room area 
RH_3, Humidity in laundry room area, in % 
T4, Temperature in office room, in Celsius 
RH_4, Humidity in office room, in % 
T5, Temperature in bathroom, in Celsius 
RH_5, Humidity in bathroom, in % 
T6, Temperature outside the building (north side), in Celsius 
RH_6, Humidity outside the building (north side), in % 
T7, Temperature in ironing room , in Celsius 
RH_7, Humidity in ironing room, in % 
T8, Temperature in teenager room 2, in Celsius 
RH_8, Humidity in teenager room 2, in % 
T9, Temperature in parents room, in Celsius 
RH_9, Humidity in parents room, in % 
To, Temperature outside (from Chievres weather station), in Celsius 
Pressure (from Chievres weather station), in mm Hg 
RH_out, Humidity outside (from Chievres weather station), in % 
Wind speed (from Chievres weather station), in m/s 
Visibility (from Chievres weather station), in km 
Tdewpoint (from Chievres weather station), Â°C 
rv1, Random variable 1, nondimensional 
rv2, Random variable 2, nondimensional 
```


#### 2、RECS

待分析

