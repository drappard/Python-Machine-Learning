#Python Machine Learning tutorial
#Part 4 - Forecasting and Predicting

########################### PREVIOUS CODE ######################################
import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

###############################################################################

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] #contains the most recent features
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print(confidence)

#array of forecasts
forecast_set = clf.predict(X_lately)
print(forecast_set, confidence, forecast_out)
#[832.03722003 840.7831898  839.87258172 835.29506997 832.96749259
# 815.88502295 791.90027576 820.18177804 817.94326729 822.83902643
# 796.27366985 823.12101028 837.14451635 826.83258634 832.25851356
# 824.22236806 833.33510879 821.12302106 822.70367366 842.77063342
# 870.86348981 858.11300596 860.40177571 838.91372777 855.77969345
# 847.84593311 870.04159377 857.83819239 884.59865272 863.65659988
# 860.85552482 864.45099001 878.92718335 877.47456346 824.55400079] 0.8941777404632077 35

#import statements to visualize the data
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
df['forecast'] = np.nan #set to NaN first
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 #seconds
next_unix = last_unix + one_day

#add the forecast to existing dataframe via iterating through the forecast set
for i in forecast_set:
      next_date = datetime.datetime.fromtimestamp(next_unix)
      next_unix += 86400
      df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]

#graph the historical data and future predictions
df['Adj. Close'].plot()
df['forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
