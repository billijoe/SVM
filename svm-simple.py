from sklearn import svm
x = [[1, 1], [2, 0], [2, 3]]
y = [0, 0, 1]
clf = svm.SVC(kernel= 'linear')
clf.fit(x, y)
w = clf.coef_[0]
# print(clf)
#
# print(clf.support_vectors_)
# print('________________')
# print(clf.support_)
# print('_______________')
# print(clf.n_support_)
# print('____________')
# print(w[0], w[1])

b = clf.support_vectors_[-1]
print(b[1])