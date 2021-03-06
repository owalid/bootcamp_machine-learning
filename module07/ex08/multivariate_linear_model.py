import numpy as np
import matplotlib.pyplot as plt

class MyLinearRegression:
  def __init__(self,  thetas, alpha=0.00001, max_iter=140000):
    self.alpha = alpha
    self.max_iter = max_iter
    self.thetas = thetas
  
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

  def plot(self, x, y, x_label, y_label, label_predicted, label_normal):
    y_hat = self.predict_(x)
    plt.grid(color='grey', linestyle='-', linewidth=0.5)
    plt.scatter(x, y, c='g', label=label_normal)
    plt.legend()
    plt.scatter(x, y_hat, c='r', s=5, label=label_predicted)
    plt.legend()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()

  def plot_multivariate(self, X, x, y, x_label, y_label, label_predicted, label_normal):
    y_hat = self.predict_(X)
    plt.grid(color='grey', linestyle='-', linewidth=0.5)
    plt.scatter(x, y, c='g', label=label_normal)
    plt.legend()
    plt.scatter(x, y_hat, c='r', s=5, label=label_predicted)
    plt.legend()
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()