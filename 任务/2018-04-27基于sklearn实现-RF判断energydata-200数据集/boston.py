from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
#Load boston housing dataset as an example
boston = load_boston()

X = boston["data"]


print(X)

Y = boston["target"]

# Y = ['60', '60', '50', '50', '60', '50', '60', '60', '60', '70', '230', '580']
print(Y)

names = boston["feature_names"]
print(names)




rf = RandomForestRegressor()
rf.fit(X, Y)
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names),
             reverse=True))