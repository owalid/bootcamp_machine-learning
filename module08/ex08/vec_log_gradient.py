import numpy as np
from sigmoid import sigmoid_

def add_intercept(x):
  return np.c_[np.ones((x.shape[0], 1)), x]

def vec_log_gradient(x, y, theta):
  x_pr = add_intercept(x)
  m = y.shape[0]
  sig = sigmoid_(np.dot(x_pr, theta))
  return (1 / m) * (np.dot(x_pr.T, (sig - y)))