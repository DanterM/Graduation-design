#coding : utf8
#python 3.5
#数据引导

import pandas as pd

def load_dataset():
	data = pd.read_csv('./data/data.csv')
	columns = data.columns
	columns = [item for item in columns if item != 'TOTKWH']
	feature, label = data[columns], data['TOTKWH']
	return feature, label
