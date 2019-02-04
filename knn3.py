#Python Machine Learning Tutorial
#Part 18 - Testing our K Nearest Neighbors Classifier

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter
import pandas as pd
import random
style.use('fivthirtyeight')

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than the total voting groups!')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance,group])
    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?',-99999,inplace=True)
df.drop(['id'],1,inplace=True)
#convert data to list of list and all float values
full_data = df.astype(float).values.tolist()

random.shuffle(full_data)

test_size = 0.2
#prepare dictionaries for training and testing to be populated
#dictioanries have two keys: 2 = benign tumors, 4 = malinant tumors
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
#select first 80% as training data
train_data = full_data[:-int(test_size*len(full_data))]
#select last 20% as testing data
test_data = full_data[-int(test_size*len(full_data)):]

#populate dictionaries, key is the class, values are the attributes
for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

#iterate through both classes/keys in dictionary
for group in test_set:
    #iterate through each datapoint in each class
    for data in test_set[group]:
        vote = k_nearest_neighbors(train_set, data, k=5)
        if group == vote
            correct += 1
        total += 1
print('Accuracy:', correct/total)
