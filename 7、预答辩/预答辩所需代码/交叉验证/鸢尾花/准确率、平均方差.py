import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
iris.data.shape, iris.target.shape((150, 4), (150,))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
X_train.shape, y_train.shape((90, 4), (90,))
X_test.shape, y_test.shape((60, 4), (60,))
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)



