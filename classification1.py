#Python Machine Learning Tutorial
#Part 14 - Applying K Nearest Neighbors to Data

#Breast Cancer Dataset - https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/

import numpy as np
from sklearn import preprocessing, cross_validation, Neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconson.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'],1,inplace=True)
