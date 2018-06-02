#coding : utf8
#python 3.5
#rf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
# from score import RMSE, R2, MAE, MAEP
# from data import load_dataset
# from precoss import decompose
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def load_dataset():
	data = pd.read_csv('./data/data.csv')
	columns = data.columns
	columns = [item for item in columns if item != 'TOTKWH']
	feature, label = data[columns], data['TOTKWH']
	return feature, label


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




house_char = ['HOUSING TYPING', 'CENCDD65', 'CENHDD65', 'UR', 'CENYEARMADE', 'CENTOTSQFT', 'WALLTYPE', 'CRAWL', 'DRAFTY']
householder_char = ['INCOME', 'EDUCATION', 'RACE', 'CENMENBER', 'ATHOME']
adoptions_char = ['THERMHEAT', 'AGEHEAT', 'OTHHEAT', 'SIZEHEAT', 'CENH2O', 'USEAC', 'TYPEAC', 'THERMAC']
environment_char = ['CENPDPI', 'CENPRICES', 'DIVISION GROUP']

def decompose(feature):
	house = feature[house_char]
	householder = feature[householder_char]
	adoptions = feature[adoptions_char]
	environment = feature[environment_char].values.tolist()
	pca = PCA(n_components='mle', svd_solver='full') #令PCA自行选择最好的压缩方式

	#environment feature decompose
	pca.fit(environment)
	new_environment = pca.transform(environment).tolist()

	#house type decompose
	pca.fit(house)
	new_house = pca.transform(house).tolist()
	#householder characteristics decompose
	pca.fit(householder)
	new_householder = pca.transform(householder).tolist()
	#adoptions of end-use appliances decompse
	pca.fit(adoptions)
	new_adoptions = pca.transform(adoptions).tolist()
	#house feature decompose
	householder_feature = []
	for item in zip(new_house, new_householder, new_adoptions):
		temp = []
		temp.extend(item[0])
		temp.extend(item[1])
		temp.extend(item[2])
		householder_feature.append(temp)
	pca.fit(householder_feature)
	new_householder_feature = pca.transform(householder_feature).tolist()
	result = []
	for item in zip(new_householder_feature, new_environment):
		temp = []
		temp.extend(item[0])
		temp.extend(item[1])
		result.append(temp)
	return result




if __name__ == '__main__':
	feature, label = load_dataset()
	feature = decompose(feature)
	rf = RandomForestRegressor() 
	result1 = cross_val_score(rf, feature, label, scoring=RMSE)
	result2 = cross_val_score(rf, feature, label, scoring=R2)
	result3 = cross_val_score(rf, feature, label, scoring=MAE)
	result4 = cross_val_score(rf, feature, label, scoring=MAEP)
	print('RMSE:\t', sum(result1)/3)
	print('R2:\t', sum(result2)/3)
	print('MAE:\t', sum(result3)/3)
	print('MAEP:\t', sum(result4)/3)





