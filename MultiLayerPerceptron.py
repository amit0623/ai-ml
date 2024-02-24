import sys

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from matplotlib.colors import ListedColormap


class Perceptron:

    """
    eta = LR 
    n-Iter = num of iterations 

    w_ = Weights array 1D 
    b_ = Bias 

    """

    def __init__(self,eta = 0.01, n_iter=50, random_state=1) -> None:
        self.eta = eta 
        self.n_iter = n_iter 
        self.random_state = random_state

    def fit (self,X,y): 
        """
        X = {array } , shape = [n_examples, n_features ]
        y = array-loke, shape = [n_examples ]
        Initialize Weights to normal distribution 
        """ 
        randomGeneration = np.random.RandomState(self.random_state)
        self.w_ = randomGeneration.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0  # init errors to 0 

            for xi,target in zip(X,y):
                update = self.eta *(target -self.predict(xi))
                self.w_ += update*xi
                self.b_ += update 
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self,X)  : 
        return np.dot(X,self.w_) + self.b_ 
    
    def predict(self, X ) : 
        return np.where(self.net_input(X) >= 0.0, 1, 0 ) 

## download the data 
    
try : 
    s = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    df = pd.read_csv(s,header=None, encoding="utf-8")
except:
    print("URL Not REACHABLE ")

print(df.tail())

y=df.iloc[0:100, 4].values
y = np.where(y=='Iris-setosa', 0, 1 ) 

# print(y)
#Extrac t petal and sepal lenght 
X=df.iloc[0:100, [0,2]].values # extract features 1 and 2 for 100 samples.

# print(X)

# Train the Perceptron 

percept = Perceptron(eta=0.1, n_iter=10) 

percept.fit(X,y)

plt.plot(range(1,len(percept.errors_)+1),percept.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Num of update ')

plt.show()
