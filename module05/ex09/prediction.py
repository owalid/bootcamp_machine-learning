import numpy as np
from tools import add_intercept

def predict_(x, theta):
  theta = np.reshape(theta, (theta.shape[0],))
  return np.sum(add_intercept(x) * theta, axis=1) 