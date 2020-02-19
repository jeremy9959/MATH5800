import math
import numpy as np
import pandas as pd

class Logistic:

    theta = None
    num = 0

    def __init__(self, max_iterations, tolerance):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def zed(self, x):
        z = 0
        for j in range(self.num):
            z += np.exp(np.dot(x, self.theta[j])) 
        return z

    def fit(self, X, y):

        num_classes = np.unique(y).shape[0]
        self.num = num_classes

        self.theta = np.zeros((num_classes, X.shape[1]))

        for r in range(self.max_iterations):
            
            #compute z
            z = self.zed(X)

            p = np.zeros((num_classes, X.shape[0]))
            for j in range(num_classes):
                p[j] = np.exp(np.dot(X, self.theta[j])) / z

            gradient = np.zeros(self.theta.shape)
            for j in range(num_classes):
                for i in range(X.shape[0]):
                    if(y[i] == j):
                        gradient[j] += X[i]
                    gradient[j] -= np.dot(X[i], self.zed(X[i]))

            print("Gradient has been computed " + str(r) + " times.")

            self.theta = self.theta - 0.01*gradient

            if(np.amax(np.abs(gradient)) < self.tolerance):
                break
        
    def probs(self, X_test):
        z = 0
        for j in range(self.num):
            z += np.exp(np.dot(X_test, self.theta[j]))
        p = np.zeros((self.num, X_test.shape[0]))
        for j in range(self.num):
            p[j] = np.exp(np.dot(X_test, self.theta[j])) / z
        return p

    def score(self, X_test,y_test):
        error = np.abs(self.probs(X_test) - y_test)
        return 1 - (np.sum(error)/error.shape[0])

    