import numpy as np

def sigmoid_(x):
  return np.array(1 / (1 + np.exp(-x)))