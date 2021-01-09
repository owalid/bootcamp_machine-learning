import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from TinyStatistician import TinyStatistician as tst
import math

def mse_(y, y_hat):
  return np.sum(mse_elem(y, y_hat))

def mse_elem(y, y_hat):
  if len(y_hat.shape) == 1:
    y_hat = y_hat[:, np.newaxis]
  if len(y.shape) == 1:
    y = y[:, np.newaxis]
  m = y.shape[0]
  return (1 / m) * np.sum((y_hat - y)**2, axis=1)

def rmse_(y, y_hat):
  return math.sqrt(mse_(y, y_hat))

def rmse_elem(y, y_hat):
  if len(y_hat.shape) == 1:
    y_hat = y_hat[:, np.newaxis]
  if len(y.shape) == 1:
    y = y[:, np.newaxis]
  m = y.shape[0]
  return np.sqrt((1 / m) * np.sum((y_hat - y)**2, axis=1))

def mae_(y, y_hat):
  return np.sum(mae_elem(y, y_hat))

def mae_elem(y, y_hat):
  if len(y_hat.shape) == 1:
    y_hat = y_hat[:, np.newaxis]
  if len(y.shape) == 1:
    y = y[:, np.newaxis]
  m = y.shape[0]
  return (1 / m) * np.sum(abs(y_hat - y), axis=1)

def r2score_(y, y_hat):
  return np.sum(r2score_elem(y, y_hat))

def r2score_elem(y, y_hat):
  if len(y_hat.shape) == 1:
    y_hat = y_hat[:, np.newaxis]
  if len(y.shape) == 1:
    y = y[:, np.newaxis]
  m = y.shape[0]
  y_mean = tst().mean(y)
  return 1 - (np.sum((y_hat - y)**2, axis=1) / np.sum((y - y_mean)**2, axis=1))


x = np.array([0, 15, -9, 7, 12, 3, -21])
y = np.array([2, 14, -13, 5, 12, 4, -19])

# Mean squared error
print("Mean squared error")
## my implementation
print("my:", mse_(x,y))
## sklearn implementation
print("sklearn:", mean_squared_error(x,y))

# Root mean squared error
print("\nRoot mean squared error")
## my implementation
print("my:", rmse_(x,y))
## sklearn implementation not available: take the square root of MSE
print("sklearn:", math.sqrt(mean_squared_error(x,y)))

# Mean absolute error
print("\nMean absolute error")
## my implementation
print("my:", mae_(x,y))
## sklearn implementation
print("sklearn:", mean_absolute_error(x,y))


# R2-score
print("\nR2-score")

## my implementation
print("my:", r2score_(x,y))

## sklearn implementation
print("sklearn:", r2_score(x,y))