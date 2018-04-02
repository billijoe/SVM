from __future__ import print_function
import logging
from time import time
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA

from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

#人脸数据下载
#选择超过70张图片的作为lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
#数据预处理
n_samples, h, w = lfw_people.images.shape
# print(n_samples)
# print(lfw_people)
# print(h)
# print(w)

X = lfw_people.data
# print(X)
n_features = X.shape[1]
print(n_features)
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
# print(X)
# print(target_names)
# print(n_classes)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25
)

##############################
#pca降维
n_components = 150
print('将' + str(n_features) + '维度转化为' + str(n_components) + ':')
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
print('完成时间:'+str(time() - t0))

#提供特征脸-将数字（一串数字）转化为‘方形’
eigenfaces = pca.components_.reshape((n_components, h, w))

print('获取若干主要成分')
t0 = time()
#获得主成分
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print('完成时间' + str(time() - t0))

##################
#svm分类器
print('将提取的特征值进行分类')
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              "gamma": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
              }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print('完成时间' + str(time() - t0))
print('最佳函数组合')
print(clf.best_estimator_)

###############
#准确度分析
print('预测测试集的人脸数据')
print('准确度分析')
t0 = time()
y_pred = clf.predict(X_test_pca)
print('完成时间' + str(time() - t0))
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

prediction_titles = [title(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]

plot_gallery(X_test, prediction_titles, h, w)

# plot the gallery of the most significative eigenfaces

eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)

plt.show()