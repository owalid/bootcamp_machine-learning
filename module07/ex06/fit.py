import numpy as np
from gradient import gradient

def fit_(x, y, theta, alpha, n_cycles):
  for _ in range(n_cycles):
    theta -= alpha * gradient(x, y, theta)
  return theta