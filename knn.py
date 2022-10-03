from tkinter import N
import numpy as np
from sklearn import preprocessing, svm,neighbors  
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

#df = pd.read_csv('spama.data.txt')
df = pd.read_csv('leak.data.txt')
df = df.replace(r'^\s*$', np.nan, regex=True)

X= np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
print(X.shape)
#print(X_train[0])
#print(y_train.shape)
#print(y_train)
plt.figure()
plt.scatter(X[:,0], X[:,1], c=y, edgecolor='k', cmap=cmap, s=40)
plt.show()


clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy= clf.score(X_test, y_test)
print(accuracy)

#example_measures = np.array([4,2,1,1,1,2,3,2,1,2])
#example_measures = example_measures.reshape(1,-1)
#prediction = clf.predict(example_measures)
#print(prediction)



