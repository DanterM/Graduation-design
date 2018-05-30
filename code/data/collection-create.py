#数据加工
#recs2009_public.csv -> data.csv
import pandas as pd

if __name__ == '__main__':
	data = pd.read_csv('./recs2009_public.csv')
	totkwh = data['KWH'].tolist()
	divisiongroup = data['DIVISION'].tolist()
	division = list(set(data['DIVISION']))
	housingtyping = data['TYPEHUQ'].tolist()
	ur = data['UR'].tolist()
	walltype = data['WALLTYPE'].tolist()
	crawl = data['CRAWL'].tolist()
	drafty = data['DRAFTY'].tolist()
	konwrent = data['KOWNRENT'].tolist()
	education = data['EDUCATION'].tolist()
	race = data['Householder_Race'].tolist()
	athome = data['ATHOME'].tolist()
	thermheat = data['AUTOHEATDAY'].tolist()
	ageheat = data['EQUIPAGE'].tolist()
	othheat = data['HEATOTH'].tolist()
	sizeheat = data['EQMAMT'].tolist()
	useac = data['USECENAC'].tolist()
	typeac = data['COOLTYPE'].tolist()
	moneypy_avarage = []
	cdd65_avarage = []
	hdd65_avarage = []
	yearmade_avarage = []
	totsqft_avarage = []
	nhsldmen_avarage = []
	h2o_avarage = []
	dollarel_avarage = []
	for index in division:
		item = data[data.DIVISION == index]['MONEYPY']
		moneypy_avarage.append(sum(item)/len(item))
		item = data[data.DIVISION == index]['CDD65']
		cdd65_avarage.append(sum(item)/len(item))
		item = data[data.DIVISION == index]['HDD65']
		hdd65_avarage.append(sum(item)/len(item))
		item = data[data.DIVISION == index]['YEARMADE']
		yearmade_avarage.append(sum(item)/len(item))
		item = data[data.DIVISION == index]['TOTSQFT']
		totsqft_avarage.append(sum(item)/len(item))
		item = data[data.DIVISION == index]['NHSLDMEM']
		nhsldmen_avarage.append(sum(item)/len(item))
		item = data[data.DIVISION == index][['NUMH2ONOTNK', 'NUMH2OHTRS']]
		h2o_avarage.append(sum(item['NUMH2ONOTNK']+item['NUMH2OHTRS'])/len(item))
		item = data[data.DIVISION == index]['DOLLAREL']
		dollarel_avarage.append(sum(item)/len(item))
	avarage = dict(zip(division, moneypy_avarage))
	cenpdpi = [avarage[i] for i in divisiongroup]
	avarage = dict(zip(division, cdd65_avarage))
	cencdd65 = [avarage[i] for i in divisiongroup]
	avarage = dict(zip(division, hdd65_avarage))
	cenhdd65 = [avarage[i] for i in divisiongroup]
	avarage = dict(zip(division, yearmade_avarage))
	cenyearmade = [avarage[i] for i in divisiongroup]
	avarage = dict(zip(division, totsqft_avarage))
	centotsqft = [avarage[i] for i in divisiongroup]
	avarage = dict(zip(division, nhsldmen_avarage))
	cenmenber = [avarage[i] for i in divisiongroup]
	avarage = dict(zip(division, h2o_avarage))
	cenh2o = [avarage[i] for i in divisiongroup]
	avarage = dict(zip(division, dollarel_avarage))
	cenprices = [avarage[i] for i in divisiongroup]
	thermac = []
	for item in data[['AIRCOND', 'PROTHERMAC']].values:
		if not item[0]:
			thermac.append(0)
		elif item[1] == 0:
			thermac.append(1)
		elif item[1] == 1:
			thermac.append(2)
		else:
			thermac.append(3)
	income = []
	for item in data['MONEYPY'].tolist():
		if item in [1, 2, 3, 4, 5, 6, 7, 8]:
			income.append(0)
		elif item in [9, 10, 11, 12, 13, 14]:
			income.append(1)
		else:
			income.append(2)
	temp = []
	for item in ur:
		if item == 'U':
			temp.append(0)
		else:
			temp.append(1)
	ur = temp
	
	result = {}
	
	result['CENPDPI'] = cenpdpi
	
	result['CENPRICES'] = cenprices

	result['DIVISION GROUP'] = divisiongroup

	result['CENCDD65'] = cencdd65

	result['CENHDD65'] = cenhdd65

	result['HOUSING TYPING'] = housingtyping

	result['UR'] = ur

	result['CENYEARMADE'] = cenyearmade

	result['CENTOTSQFT'] = centotsqft

	result['WALLTYPE'] = walltype

	result['CRAWL'] = crawl

	result['DRAFTY'] = drafty

	result['KONWRENT'] = konwrent

	result['INCOME'] = income
	
	result['EDUCATION'] = education
	
	result['RACE'] = race

	result['CENMENBER'] = cenmenber

	result['ATHOME'] = athome

	result['THERMHEAT'] = thermheat

	result['AGEHEAT'] = ageheat
	
	result['OTHHEAT'] = othheat
	
	result['SIZEHEAT'] = sizeheat

	result['CENH2O'] = cenh2o

	result['USEAC'] = useac

	result['TYPEAC'] = typeac
	
	result['THERMAC'] = thermac
	
	result['TOTKWH'] = totkwh

	pd.DataFrame(result).to_csv('./data.csv', index=False)
