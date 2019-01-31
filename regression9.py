#Python Machine Learning tutorial
#Part 9 - Creating Sample Data

#Mathematically we know the R squared value will be lower when the best-fit line is worse, and higher (closer to 1), when the best fit line is better. The goal is to generate a sample dataset where we can dictate the parameters to either improve or decrease the R squard value and the best fit line

from statistics import mean
import numpy as np
import random #to generate random datasets
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#hm - how many data points we want in a set
#variance - amount that dictates how far each data point can be from the previous data point
#step - how far to step on average per point, default is 2
#correlation - options are False, positive (pos), or negative (neg)
def create_dataset(hm,variance,step=2,correlation=False):
    val = 1
    ys = []
    #iterate through the range and generate sample y values by appending the current value plus a random range of the negative variance to positive variance
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    #create sample x values
    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64),np.array(ys,dtype=np.float64)

#best fit line code from previous section
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))

    b = mean(ys) - m*mean(xs)

    return m, b

#coefficient of determination code from previous section - to check for the accuracy and reliability of our best-fit line
def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]

    squared_error_regr = sum((ys_line - ys_orig) * (ys_line - ys_orig))
    squared_error_y_mean = sum((y_mean_line - ys_orig) * (y_mean_line - ys_orig))

    print(squared_error_regr)
    print(squared_error_y_mean)

    r_squared = 1 - (squared_error_regr/squared_error_y_mean)

    return r_squared

#generates the random sample dataset
xs, ys = create_dataset(40,40,2,correlation='pos')
#xs, ys = create_dataset(40,10,2,correlation='pos') #lower variance to improve tightness of correlation
#xs, ys = create_dataset(40,10,2,correlation='neg') #negative correlation
#xs, ys = create_dataset(40,10,2,correlation=False) #remove correlation
m, b = best_fit_slope_and_intercept(xs,ys)
regression_line = [(m*x)+b for x in xs]
r_squared = coefficient_of_determination(ys,regression_line)
#higher r squared value = better best fit line
print(r_squared)

#plot the dataset with regressionline
plt.scatter(xs,ys,color='#003F72', label = 'data')
plt.plot(xs, regression_line, label = 'regression line')
plt.legend(loc=4)
plt.show()

#Output
#First Run: 22899.217729831144, 46633.975000000006, 0.5089584850995195
