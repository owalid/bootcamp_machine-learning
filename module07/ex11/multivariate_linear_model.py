import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegression:
  def __init__(self,  thetas, alpha=0.00001, max_iter=140000):
    self.alpha = alpha
    self.max_iter = max_iter
    self.thetas = thetas.astype(np.float32)
  
  @staticmethod
  def add_intercept(x):
    return np.c_[np.ones((x.shape[0], 1)), x]

  @staticmethod
  def gradient(self, x, y):
    x_pr = self.add_intercept(x)
    m = y.shape[0]
    return (1 / m) * np.dot(np.transpose(x_pr), (np.dot(x_pr, self.thetas) - y))
  
  def predict_(self, x):
    return np.dot(self.add_intercept(x), self.thetas)

  def fit_(self, x, y):
    for _ in range(self.max_iter):
      self.thetas -= self.alpha * self.gradient(self, x, y)
    return self.thetas

  def cost_elem_(self, x, y):
    m = y.shape[0]
    y_hat = self.predict_(x)
    return (1 / (2 *m)) * (y_hat - y)**2
  
  def cost_(self, x, y):
    return np.sum(self.cost_elem_(x, y))

  def mse_(self, x, y):
    y_hat = self.predict_(x)
    y, y_hat = np.array(y), np.array(y_hat)
    return np.square(np.subtract(y,y_hat)).mean() 
