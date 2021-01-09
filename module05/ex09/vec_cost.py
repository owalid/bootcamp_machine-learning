import numpy as np

def cost_(y, y_hat):
  m = y.shape[0]
  # sum_yy = 0
  # for i in range(m):
  #   sum_yy += (y_hat[i] - y[i])**2
  return (1 / (2 *m)) * np.sum((y_hat - y)** 2)
