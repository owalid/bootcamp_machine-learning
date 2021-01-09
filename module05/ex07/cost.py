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

x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
theta1 = np.array([[2.], [4.]])
y_hat1 = predict_(x1, theta1)
y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

# Example 1:
print(cost_elem_(y1, y_hat1))
print(cost_(y1, y_hat1))


x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
theta2 = np.array([[0.05], [1.], [1.], [1.]])
y_hat2 = predict_(x2, theta2)
y2 = np.array([[19.], [42.], [67.], [93.]])

print(cost_elem_(y2, y_hat2))
print(cost_(y2, y_hat2))


x3 = np.array([0, 15, -9, 7, 12, 3, -21])
theta3 = np.array([[0.], [1.]])
y_hat3 = predict_(x3, theta3)
y3 = np.array([2, 14, -13, 5, 12, 4, -19])

print(cost_(y3, y_hat3))
print(cost_(y3, y3))
