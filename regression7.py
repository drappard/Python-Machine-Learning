#Python Machine Learning tutorial
#Part 7 - How to Program the Best Fit Line Pt. 2

########################### PREVIOUS CODE ######################################
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))

    b = mean(ys) - m*mean(xs)

    return m, b

regression_line = [(m*x)+b for x in xs]
################################################################################

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

predict_x = 7
predict_y = (m*predict_x)+b
print(predict_y)

plt.scatter(xs,ys,color='#003F72',label='data')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()
