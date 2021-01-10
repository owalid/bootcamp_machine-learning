import numpy as np
from prediction import predict_

def simple_gradient(x, y, theta):
  y_hat = predict_(x, theta)
  m = x.shape[0]
  nabla_j0 = theta[0] - ((1/m) * np.sum((y_hat - y)))
  nabla_j1 = theta[1] - ((1/m) * np.sum((y_hat - y) * x))
  return [nabla_j0, nabla_j1]

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
theta1 = np.array([2, 0.7])
print(simple_gradient(x, y, theta1))


theta2 = np.array([1, -0.4])
print(simple_gradient(x, y, theta2))