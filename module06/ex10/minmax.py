import numpy as np

def minmax(x):
  min_x = min(x)
  max_x = max(x)
  return (x - min_x) / (max_x - min_x)

