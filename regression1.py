#Python Machine Learning tutorial
#Part 1 - Intro and Data

###############################################################################
#required installation of python packages:
#pip install numpy
#pip install scipy
#pip install scikit-learn
#pip install matplotlib
#pip install pandas
#pip install quandl
################################################################################

import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")
print(df.head())
#              Open    High     Low    Close      Volume  Ex-Dividend  Split Ratio  Adj. Open  Adj. High   Adj. Low  Adj. Close  Adj. Volume
#Date
#2004-08-19  100.01  104.06   95.96  100.335  44659000.0          0.0          1.0  50.159839  52.191109  48.128568   50.322842   44659000.0
#2004-08-20  101.01  109.08  100.50  108.310  22834300.0          0.0          1.0  50.661387  54.708881  50.405597   54.322689   22834300.0
#2004-08-23  110.76  113.48  109.05  109.400  18256100.0          0.0          1.0  55.551482  56.915693  54.693835   54.869377   18256100.0
#2004-08-24  111.24  111.60  103.57  104.870  15247300.0          0.0          1.0  55.792225  55.972783  51.945350   52.597363   15247300.0
#2004-08-25  104.76  108.00  103.88  106.000   9188600.0          0.0          1.0  52.542193  54.167209  52.100830   53.164113    9188600.0

df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
print(df.head())

#            Adj. Close    HL_PCT  PCT_change  Adj. Volume
#Date
#2004-08-19   50.322842  8.072956    0.324968   44659000.0
#2004-08-20   54.322689  7.921706    7.227007   22834300.0
#2004-08-23   54.869377  4.049360   -1.227880   18256100.0
#2004-08-24   52.597363  7.657099   -5.726357   15247300.0
#2004-08-25   53.164113  3.886792    1.183658    9188600.0
