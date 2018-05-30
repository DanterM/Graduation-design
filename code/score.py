#coding : utf8
#python 3.5
#loss funcation

import numpy as np
from sklearn.metrics import make_scorer

def r2(truch, predictions):
	average = np.sum(truch)/len(truch)
	tot = np.sum(np.square(truch-average))
	res = np.sum(np.square(truch-predictions))
	return 1-res/tot

R2 = make_scorer(r2)


def rmse(truch, predictions):
	return np.sqrt(np.sum(np.square(truch-predictions))/len(truch))

RMSE = make_scorer(rmse, greater_is_better=False)

def mae(truch, predictions):
	return np.sum(np.abs(truch-predictions))/len(truch)

MAE = make_scorer(mae, greater_is_better=False)

def maep(truch, predictions):
	return np.sum(np.abs(truch - predictions)/truch)/len(truch)

MAEP = make_scorer(maep, greater_is_better=False)
