import numpy as np

def add_intercept(x):
  return np.c_[np.ones((x.shape[0], 1)), x]

def gradient(x, y, theta):
  x_pr = add_intercept(x)
  m = y.shape[0]
  return (1 / m) * np.dot(np.transpose(x_pr), (np.dot(x_pr, theta) - y))
