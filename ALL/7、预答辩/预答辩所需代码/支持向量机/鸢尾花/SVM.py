import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets

iris=datasets.load_iris()
x=iris.data[:,:2]
y=iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=1)

model=svm.SVC(C=1, kernel='linear', decision_function_shape='ovr')
model.fit(x_train,y_train)

y_hat=model.predict(x_test)
print('训练集上的正确率为%2f%%'%( model.score(x_train, y_train)*100))
print('测试集上的正确率为%2f%%'%( model.score(x_test, y_test)*100))
