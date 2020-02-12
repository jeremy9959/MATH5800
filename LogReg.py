import math
import numpy as np
import pandas as pd

class Logistic:

    theta = None

    def __init__(self, max_iterations, tolerance):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])

        for i in range(self.max_iterations):
            z = 1 / (1 + np.exp(-np.dot(X,self.theta)))
            gradient = (1/X.shape[1])*(np.dot(X.T, (z-y)))
            self.theta = self.theta - 0.01*gradient
            if(np.amax(np.abs(gradient)) < self.tolerance):
                break
        
    def probs(self, X_test):
        return 1 / (1 + np.exp(-np.dot(X_test,self.theta)))

    def score(self, X_test,y_test):
        error = np.abs(self.probs(X_test) - y_test)
        return 1 - (np.sum(error)/error.shape[0])

    