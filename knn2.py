#Python Machine Learning Tutorial
#Part 15 - Creating a K Nearest Neighbors Classifer from scratch

import numpy as np
import matplotlib.pyplot as pyplot
from matplotlib import style
import warnings
from math import sqrt
from collections import counter
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5.7]

[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1],s=100)

plt.show()
