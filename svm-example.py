import numpy as np
import pylab as pl
from sklearn import svm

#创建40个点
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]] #维度提升
# print(X)
Y = [0]*20 + [1]*20

clf = svm.SVC(kernel= 'linear')
clf.fit(X, Y)

#建立超平面
w = clf.coef_[0] #coaf 存储w1-w...的值
k = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = k * xx - (clf.intercept_[0])/w[1] #intercept存储w0

#得到和超平面相平行的，并且与支持向量相切的点
b = clf.support_vectors_[0]
yy_down = k * xx +(b[1] - k * b[0])
b = clf.support_vectors_[-1]
yy_up = k * xx +(b[1] - k * b[0])

print('w的值', w)
print('斜率k是', k)
print('支持向量是', clf.support_vectors_)


pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=80, facecolors = 'yellowgreen')
pl.scatter(X[:, 0], X[:, 1], c = 'r', cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()