from sklearn import datasets
import pandas as pd
from sklearn import preprocessing, svm,neighbors 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
stdsc = StandardScaler()

import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
vect = TfidfVectorizer()
# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

#iris = datasets.load_iris()

#X = iris["data"][:, (2, 3)]  # petal length, petal width
#y = iris["target"]

data = pd.read_csv('iris.data.txt')
#iris =data.head()
X = data.drop(['Leak Size'], axis=1)
#print(X.shape)
y = data['Leak Size']
#print(y.shape)

X =np.array(X).reshape((-1, 1))
y =np.array(y).reshape((-1, 1))
#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)


setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# SVM Classifier model
svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X, y)
# Bad models
x0 = np.linspace(0, 5.5, 200)
pred_1 = 5*x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

plt.figure(figsize=(12,2.7))


plt.subplot(122)
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label='Iris-Setosa')
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
#plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 5.5, 0, 2])
plt.show()  
