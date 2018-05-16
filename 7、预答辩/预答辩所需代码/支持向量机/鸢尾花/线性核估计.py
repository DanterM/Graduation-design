import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm,datasets

iris = datasets.load_iris()
# print(iris)

# 只选取每组数据中的前两个
X = iris.data[:,:2]
# print(X)
y = iris.target



C = 1.0
svc = svm.SVC(kernel='linear',C=1,gamma='auto').fit(X,y)
x_min,x_max = X[:,0].min()-1,X[:,0].max()+1
y_min,y_max = X[:,1].min()-1,X[:,1].max()+1

xx,yy = np.meshgrid(np.arange(x_min,x_max),np.arange(y_min,y_max))

plt.subplots(1,1,1)
Z = svc.predict(np.c_[xx.ravel(),yy.ravel()])




Z = Z.reshape(xx.shape)
plt.contour(xx,yy,Z,cmap=plt.cm.Paired)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.xlim(xx.min(),xx.max())
plt.title("SVC with linear kernel")
plt.show()