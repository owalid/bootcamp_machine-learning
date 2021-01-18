import numpy as np

def vec_log_loss_(y, y_hat, eps=1e-15):
  y_hat += eps
  m = y.shape[0]
  one_v = np.ones(m).squeeze()
  y = y.squeeze()
  y_hat = y_hat.squeeze()
  return (- 1 / m) * np.sum(((y * np.log(y_hat)) + (one_v - y) * (np.log(one_v - y_hat))))