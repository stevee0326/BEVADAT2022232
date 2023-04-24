import numpy as np

class LinearRegression:
    def __init__(self, epochs: int = 1000, lr: float = 1e-3):
        self.epochs = epochs
        self.lr = lr
        self.m = 0 #meredekség
        self.c = 0 #x metszet

    def fit(self, X: np.array, y: np.array):
        n = float(len(X))
        for i in range(self.epochs):
            y_pred = self.m * X + self.c    #curr y pred
            residuals = y - y_pred  #distance
            
            #derivatings
            D_m = (-2/n) * sum(X * residuals)
            D_c = (-2/n) * sum(residuals)
            
            #updatings
            self.m = self.m - self.lr * D_m
            self.c = self.c - self.lr * D_c

    def predict(self, X):
        return self.m * X + self.c
    



