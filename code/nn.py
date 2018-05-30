#coding : utf8
#python 3.5
#神经网络

import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import concatenate

from data import load_dataset
from precoss import house_char
from precoss import householder_char
from precoss import adoptions_char
from precoss import environment_char
from nnscore import rmse, mae, maep, r2

if __name__ == '__main__':
	data, label = load_dataset()
	house = data[house_char]
	householder = data[householder_char]
	adoptions = data[adoptions_char]
	environment = data[environment_char]
	
	house_input = Input(shape=(9,), name='house_input')
	x = Dense(units=9, activation='relu')(house_input)
	house_output = Dense(units=9, activation='relu')(x)
	
	householder_input = Input(shape=(5,), name='householder_input')
	x = Dense(units=5, activation='relu')(householder_input)
	householder_output = Dense(units=5, activation='relu')(x)
	
	adoptions_input = Input(shape=(8,), name='adoptions_input')
	x = Dense(units=8, activation='relu')(adoptions_input)
	adoptions_output = Dense(units=8, activation='relu')(x)
	
	environment_input = Input(shape=(3,),name='environment_input')
	x = concatenate([house_output, householder_output, adoptions_output, environment_input])
	x = Dense(units=25, activation='relu')(x)
	x = Dense(units=25, activation='relu')(x)
	output = Dense(units=1, activation='relu', name='output')(x)
	
	model = Model(inputs=[house_input, householder_input, adoptions_input, environment_input], outputs=output)
	model.compile(optimizer='sgd', loss=maep)#rmse, mae, maep, r2
	model.fit({'house_input':house, 'householder_input':householder, 'adoptions_input': adoptions, 'environment_input':environment}, {'output':label}, epochs=100000, batch_size=100)
	
	







	
