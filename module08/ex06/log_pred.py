import numpy as np

def add_intercept(x):
  return np.c_[np.ones((x.shape[0], 1)), x]

def logistic_predict_(x, theta):
  x_p = add_intercept(x)
  return np.array(1 / (1 + np.exp(np.dot(-x_p, theta))))