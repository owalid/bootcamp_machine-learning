import numpy as np
from prediction import predict_

def cost_elem_(y, y_hat):
  if len(y_hat.shape) == 1:
    y_hat = y_hat[:, np.newaxis]
  if len(y.shape) == 1:
    y = y[:, np.newaxis]
  m = y.shape[0]
  return (1 / (2 *m)) * np.sum((y_hat - y)**2, axis=1)

def cost_(y, y_hat):
  return np.sum(cost_elem_(y, y_hat))
