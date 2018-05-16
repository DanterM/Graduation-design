import numpy as np
import pandas as pd

train_df = pd.read_csv('../dataset/1-energydata-del1day.csv')

train_data = train_df['date']
print(train_data)

# for i in train_data:
#
#     print(train_data[[i]])

# print(train_df['date'][0])

# if train_df['date'][0] = 1:
#     print()

for i in train_df['date']:
    if (train_df['date'][i] == '0:00'):
        print('success')


def numbers_to_strings(argument):
    switcher = {
        0: "zero",
        1: "one",
        2: "two",
    }
    return switcher.get(argument, "nothing")