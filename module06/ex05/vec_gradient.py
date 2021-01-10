import numpy as np
from tools import add_intercept
from prediction import predict_

def gradient(x, y, theta):
  m = x.shape[0]
  Xpr = add_intercept(x)
  Xt = np.transpose(Xpr)
  y_hat = predict_(x, theta)
  return (1/m) * np.dot(Xt, (y_hat - y))