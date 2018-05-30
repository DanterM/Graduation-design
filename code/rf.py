#coding : utf8
#python 3.5
#rf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from score import RMSE, R2, MAE, MAEP
from data import load_dataset
from precoss import decompose

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
