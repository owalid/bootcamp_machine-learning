import numpy as np
from tools import add_intercept

def cost_elem_(y, y_hat):
  if len(y_hat.shape) == 1:
    y_hat = add_intercept(y_hat)
  if len(y.shape) == 1:
    y = add_intercept(y)
  m = y.shape[0]
  return (1 / (2 *m)) * np.sum((y_hat - y)**2, axis=1)

def cost_(y, y_hat):
  return np.sum(cost_elem_(y, y_hat))


X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(cost_(X, Y))