import numpy as np

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
