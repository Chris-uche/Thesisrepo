# Step 1
# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score 

# Step 2
# Loading the dataset
#dataset = load_digits()
df = pd.read_csv('leak.data.txt')

X= np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=4, test_size=0.2)

# Step 3
# Splitting the data into tst and train
# 80 - 20 Split
#x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.20, random_state=4)

# Step 4
# Making the Neural Network Classifier
NN = MLPClassifier()

# Step 5
# Training the model on the training data and labels
NN.fit(X_train, y_train)

# Step 6
# Testing the model i.e. predicting the labels of the test data.
y_pred = NN.predict(X_test)

# Step 7
# Evaluating the results of the model
accuracy = accuracy_score(y_test,y_pred)*100
confusion_mat = confusion_matrix(y_test,y_pred)

# Step 8
# Printing the Results
print("Accuracy for Neural Network is:",accuracy)
print("Confusion Matrix")
#print(confusion_mat)
