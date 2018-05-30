#coding : utf8
#python 3.5


import pandas as pd
import keras
import time as te
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM

def data_load_process(filePath):
	data = pd.read_csv(filePath)
	columns = sorted(data.columns.tolist())
	data, label = data[columns[1:]], data[columns[0]]
	time = data['date'].values.tolist()
	data = data.drop(columns='date')
	toint = lambda item : te.mktime(te.strptime(item, '%Y-%m-%d %H:%M:%S'))
	new_time = list(map(toint, time))
	data['date'] = pd.Series(new_time, index=data.index)
	return data, label
	
	

if __name__ == '__main__':
	data, label = data_load_process('./data/energydata_complete.csv')
	#data_len = len(data)	

	data_input = Input(shape=(28,), name='data_input')
	x = Dense(units=56, activation='relu')(data_input)
	#x = LSTM(units=56, input_dim=(data_len, 10, 28))(x)
	x = Dense(units=28, activation='relu')(x)
	x = Dense(units=14, activation='relu')(x)
	x = Dense(units=7, activation='relu')(x)
	label_output = Dense(units=1, activation='relu', name='label_output')(x)
	
	model = Model(inputs=data_input, outputs=label_output)
	model.compile(optimizer='adam', loss='msle')
	model.fit({'data_input':data}, {'label_output':label}, epochs=100000, batch_size=100)



	
	
