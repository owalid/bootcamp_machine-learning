import numpy as np
import matplotlib.pyplot as plt
# from cost import cost_elem_, cost_
from vec_cost import cost_
from prediction import predict_

def plot_with_cost(x, y, theta):
  y_hat = predict_(x, theta)
  # const_elm = cost_elem_(y, y_hat)
  plt.title(f'Cost: {cost_(y, y_hat)}')
  plt.scatter(x, y)
  plt.plot(x, y_hat, c='r')
  plt.show()


x = np.arange(1,6)
y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])

# Example 1:
theta1= np.array([18,-1])
plot_with_cost(x, y, theta1)