#Python Machine Learning tutorial
#Part 3 - Training and Testing

########################### PREVIOUS CODE ######################################
import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")

print(df.head())
print(df.tail())

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

###############################################################################

#drop any NaN (Not a Number) data in the dataframe
df.dropna(inplace=True)
#define fatures (x) and label (y)
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
#preprocesssing scales features between -1 and 1
X = preprocessing.scale(X)
y = np.array(df['label'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#use support vector regression classifier
clf = svm.SVR()
#clf = LinearRegression()
#train classifier - first with SVR, second with LR, third with threading
clf.fit(X_train, y_train)
#SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
#  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
#Linear regression using threading, -1 = all available threads
#clf = LinearRegression(n_jobs=-1)

confidence = clf.score(X_test, y_test)
print(confidence)
#SVR confidence = 0.797030379330396 -> (0.960075071072 = tutorial %)
#Linear Reggression confidence = 0.9748193577416013

#SVR kernel for transforming your data to run faster
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print(k,confidence)

#Output
#SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
#  kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
#	linear 0.9721318367073734

#SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
#  kernel='poly', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
#poly 0.5148561753935299

#SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
#  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
#rbf 0.7748923550624778

#SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
#  kernel='sigmoid', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
#sigmoid 0.8743222943924868
