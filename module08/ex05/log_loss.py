import numpy as np

def log_loss_(y, y_hat, eps=1e-15):
  m = y.shape[0]
  y_hat += eps
  return (-1 / m) * (np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))