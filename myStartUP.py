import sklearn
import numpy
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn import preprocessing
import skimage
from skimage import morphology

digits = datasets.load_digits()
print(digits.data)
print(digits.targets)
print(digits.images[0])
clf = svm.SVC
clf = svm.SVC(gamma=0.001,C=1) 
clf.verbose=1
x,y=digits.data[:-1],digits.target[:-1]

X_scaled = preprocessing.scale(x)  #try to use scaling
X_scaled.mean()
X_scaled.std()

clf.fit(x,y)
clf.gamma
clf.n_support_
clf.support_vectors_
clf.support_vectors_[0]
clf.support_
clf.classes_
clf.dual_coef_[0].size
print('Prediction:',clf.predict(digits.data[:-1]))
plt.imshow(digits.images[-1],cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
clf.get_params(deep=True)
pr = clf.predict