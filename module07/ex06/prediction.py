import numpy as np

def add_intercept(x):
  return np.c_[np.ones((x.shape[0], 1)), x]

def predict_(x, theta):
  return np.dot(add_intercept(x), theta)