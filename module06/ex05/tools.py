import numpy as np

def add_intercept(x):
  return np.c_[np.ones((x.shape[0], 1)), x]