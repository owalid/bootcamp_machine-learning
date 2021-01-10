import numpy as np
from tools import add_intercept
from prediction import predict_

def gradient(x, y, theta):
  m = x.shape[0]
  Xpr = add_intercept(x)
  Xt = np.transpose(Xpr)
  return theta - (1/m) * np.dot(Xt, (np.dot(Xpr, theta) - y))

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])

# Example 0:
theta1 = np.array([2, 0.7])
print(gradient(x, y, theta1))