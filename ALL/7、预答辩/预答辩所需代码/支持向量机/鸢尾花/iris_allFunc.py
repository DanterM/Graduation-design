import numpy as np
import mlpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:windowsfontssimsun.ttc",size=14)
iris = np.loadtxt('iris.csv',delimiter=',')
fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')
ax.scatter(iris[:,1],iris[:,2],iris[:,3],c=iris[:,4])
ax.set_xlabel("sepal length")
ax.set_ylabel("sepal width")
ax.set_zlabel("petal length")
ax.set_title("Fisher's Iris Data")
plt.show()



x,y = iris[:,:4], iris[:,4].astype(np.int)
pca = mlpy.PCA()
pca.learn(x)
z = pca.transform(x,k=2)
fig2 = plt.figure(2)
plt.scatter(z[:,0],z[:,1],c=y)
plt.title(u"主成分分析法（PCA）降维后的数据",fontproperties=font)
plt.xlabel(u"第一主成分量",fontproperties=font)
plt.ylabel(u"第二主成分量",fontproperties=font)
plt.show()





linear_svm = mlpy.LibSvm(kernel_type='linear')
linear_svm.learn(z,y)
xmin,xmax = z[:,0].min()-0.1, z[:,0].max()+0.1
ymin,ymax = z[:,1].min()-0.1, z[:,1].max()+0.1
xx,yy = np.meshgrid(np.arange(xmin,xmax,0.01),np.arange(ymin,ymax,0.01))
zgrid = np.c_[xx.ravel(),yy.ravel()]
yp = linear_svm.pred(zgrid)
fig3 = plt.figure(3)
plt.title(u"支撑向量机分类-线性核函数",fontproperties=font)
plot1 = plt.pcolormesh(xx,yy,yp.reshape(xx.shape))
plot2 = plt.scatter(z[:,0],z[:,1],c=y)
plt.xlabel(u"第一主成分量",fontproperties=font)
plt.ylabel(u"第二主成分量",fontproperties=font)
plt.xlim(xmin,xmax)
plt.ylim(ymin,ymax)
plt.show()