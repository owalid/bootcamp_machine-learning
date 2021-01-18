import numpy as np

class MyLogisticRegression():
  def __init__(self, thetas, alpha=1e-4, max_iter=100000):
    self.alpha = alpha
    self.max_iter = max_iter
    self.thetas = np.array(thetas)

  @staticmethod
  def add_intercept(x):
    return np.c_[np.ones((x.shape[0], 1)), x]

  @staticmethod
  def sigmoid_(x):
    return np.array(1 / (1 + np.exp(-x)))

  @staticmethod
  def gradient_(self, x, y):
    x_pr = self.add_intercept(x)
    m = y.shape[0]
    print(self.thetas)
    sig = self.sigmoid_(np.dot(x_pr, self.thetas))
    print(sig)
    return (1 / m) * np.dot(x_pr.T, (sig - y))

  def fit_(self, x, y):
    self.gradient_(self, x, y)
    # for _ in range(self.max_iter):
    #   self.thetas =
    return self.thetas

  def predict_(self, x):
    x_p = self.add_intercept(x)
    return np.array(1 / (1 + np.exp(np.dot(-x_p, self.thetas))))

  def cost_(self, x, y, eps=1e-15):
    y_hat = self.predict_(x)
    y_hat += -eps
    m = y.shape[0]
    one_v = np.ones(m).squeeze()
    y = y.squeeze()
    y_hat = y_hat.squeeze()
    return (- 1 / m) * np.sum(((y * np.log(y_hat)) + (one_v - y) * (np.log(one_v - y_hat))))
