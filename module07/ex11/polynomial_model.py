import numpy as np

def add_polynomial_features(x, power):
  result = np.repeat(x, power, axis=1)
  for i in range(power):
    result[:, i] = np.power(result[:, i], i + 1)
  return result