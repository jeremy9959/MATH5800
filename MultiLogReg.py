import math
import numpy as np
import pandas as pd

class Logistic:

    theta = None
    num = 0
    z = 0

    def __init__(self, max_iterations, tolerance):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def zed(self, x):
        z = np.zeros(x.shape[0])
        for j in range(self.num):
            z += np.exp(np.dot(x, self.theta[j])) 
        return z

    def p(self, x):
        p = np.zeros((self.num, x.shape[0]))
        for j in range(self.num):
            p[j] = np.exp(np.dot(x, self.theta[j])) / self.zed(x)
        return p


    def fit(self, X, y):

        num_classes = np.unique(y).shape[0]
        self.num = num_classes

        self.theta = np.zeros((num_classes, X.shape[1]))

        for r in range(self.max_iterations):


            gradient = np.zeros(self.theta.shape)
            for j in range(num_classes):
                for i in range(X.shape[0]):
                    if(y[i] == j):
                        gradient[j] += X[i]
                    gradient[j] -= X[i] * np.exp(np.dot(X[i],self.theta[j]) / self.zed(X[i]))


            #print(self.theta[0])
            gradient *= (1/X.shape[0])
            self.theta = self.theta + 0.01*gradient

            if(np.amax(np.abs(gradient)) < self.tolerance):
                break
        

    def score(self, X_test,y_test):

        probs = self.p(X_test)

        count = 0
        for i in range(X_test.shape[0]):
            if np.where(probs.T[i] == np.amax(probs.T[i])) == y_test[i]:
                count += 1

        return count / X_test.shape[0]

    