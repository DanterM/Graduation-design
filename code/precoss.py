#coding : utf8
#python 3.5
#数据压缩过程

from sklearn.decomposition import PCA

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
	
	
