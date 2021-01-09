import numpy as np

def cost_(y, y_hat):
  m = y.shape[0]
  # sum_yy = 0
  # for i in range(m):
  #   sum_yy += (y_hat[i] - y[i])**2
  return (1 / (2 *m)) * np.sum((y_hat - y)** 2)

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

# Example 1:
print(cost_(X, Y))
# Output:
# 2.142857142857143
print(cost_(X, X))
# Output:
# 0
