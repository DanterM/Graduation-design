from keras import backend as K
import numpy as np
#神经网络loss函数

def r2(truch, predictions):
	average  = K.mean(truch, axis=-1)
	tot = K.sum(K.square(truch-average))
	res = K.sum(K.square(truch-predictions))
	return 1-res/tot

def rmse(truch, predictions):
	return K.mean(K.square(predictions - truch), axis=-1)

def mae(truch, predictions):
	 return K.mean(K.abs(predictions - truch), axis=-1)

def maep(truch, predictions):
	diff = K.abs((truch - predictions) / K.clip(K.abs(truch), K.epsilon(), np.inf))
	return 100. * K.mean(diff, axis=-1)
