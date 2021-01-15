import numpy as np

def cost_(y, y_hat):
  m = y.shape[0]
  return float((1 / (2 * m)) * np.sum((y - y_hat)**2))