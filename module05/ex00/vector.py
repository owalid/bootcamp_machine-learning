import numpy as np

class Vector:
  def __init__(self, arr):
    self.value = np.array(arr)
    self.shape = self.value.shape

  def __add__(self, other):
   return np.sum(self.value, other)
  
  def __radd__(self, other):
   return np.sum(self.value, other)
  
  def __sub__(self, other):
    return np.diff(self.value, other)

  def __rsub__(self, other):
    return np.diff(self.value, other)

  def __truediv__(self, other):
    return self.value / float(other)
  
  def __rtruediv__(self, other):
    return self.value / float(other)
  
  def __mul__(self, other):
    return self.value * float(other)

  def __rmul__(self, other):
    return self.value * float(other)

  def __str__(self):
    return self.value
  
  def __repr__(self):
    return self.value