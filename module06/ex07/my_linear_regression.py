import numpy as np
from minmax import minmax
import matplotlib.pyplot as plt

class MyLinearRegression():
  """
	Description:
		My personnal linear regression class to fit like a boss.
	"""
  def __init__(self,  thetas, alpha=0.001, max_iter=1000):
    self.alpha = alpha
    self.max_iter = max_iter
    self.thetas = thetas

  @staticmethod
  def add_intercept(x):
    return np.c_[np.ones((x.shape[0], 1)), x]
    
  @staticmethod
  def gradient(self, x, y):
    m = x.shape[0]
    Xpr = self.add_intercept(x)
    Xt = np.transpose(Xpr)
    y_hat = self.predict_(x)
    return (1/m) * np.dot(Xt, (y_hat - y))

  # @staticmethod
  def mse_(self, y, y_hat): 
    y,  y_hat = np.array(y), np.array(y_hat)
    return np.square(np.subtract(y,y_hat)).mean() 

  # def mse_(self, y, y_hat):
  #   return np.sum(self.mse_elem(self, y, y_hat))

  
  def fit_(self, x, y):
    for _ in range(self.max_iter):
      self.thetas -= self.alpha * self.gradient(self, x, y)
    return self.thetas
  
  def predict_(self, x):
    return np.dot(self.add_intercept(x), self.thetas)

  def cost_elem_(self, x, y):
    m = y.shape[0]
    y_hat = self.predict_(x)
    return (1 / (2 *m)) * (y_hat - y)**2
  
  def cost_(self, x, y):
    return np.sum(self.cost_elem_(x, y))

  def plot(self, x, y):
    self.fit_(x, y)
    normalize_data = np.arange(x.min(), x.max() + 1)
    plt.scatter(x, y);
    plt.plot(normalize_data, self.predict_(normalize_data))
    plt.show()