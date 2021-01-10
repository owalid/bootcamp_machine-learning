import numpy as np
from vec_gradient import gradient
from tools import add_intercept
from prediction import predict_

def fit_(x, y, theta, alpha, max_iter):
  theta = theta.astype(np.float32)
  for _ in range(max_iter):
    theta -= (gradient(x, y, theta) * alpha)
  return theta

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
theta= np.array([1, 1])

theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
print(theta1)

print(predict_(x, theta1))
