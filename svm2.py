
#Python Machine Learning Tutorial
#Part 25 Support Vector Machine introduction

import matplotlib.pyplot as plt
from matplotlib import style
import numbpy as np
style.use('ggplot')

data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),

              1:np.array([5,1],
                         [6,-1],
                         [7,3],])}

class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)

        #train
        def fit(self,features):
            self.data = data_dict
            #{ ||w||: [w,b] }
            opt_dict = {}

            transforms = [[1,1],
                          [-1,1],
                          [-1,-1],
                          [1,-1]]


        def predict(self,features):
            #sign(x.w+b)
            classification = np.sign(np.dot(np.array(features),self.w)+self.b)
