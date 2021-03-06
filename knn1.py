#Python Machine Learning Tutorial
#Part 14 - Applying K Nearest Neighbors to Data

#Breast Cancer Dataset - https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
#KNN Applied from Scikit-Learn package

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
#replace missing data
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)
#Accuracy Output: 0.9428571428571428

################# Addition of ID Column ##################
#Impact of meaningless/misleading data by commenting out df.drop ID
#Accuracy drops by ~30%
#Accuracy Output with ID column: 0.65

################# Testing and Prediction #################
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(2,-1)
prediction = clf.predict(example_measures)
print(prediction)

#Two Sample Output:
#Acurracy: 0.9714285714285714
#Prediction: [2 2]
