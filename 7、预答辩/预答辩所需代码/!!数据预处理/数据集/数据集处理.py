import numpy as np
import pandas as pd

values = pd.read_csv('0-energydata原版.csv')

date = values['date']
# print(date)

date = np.delete(date,(2),axis=0)
print(date)

