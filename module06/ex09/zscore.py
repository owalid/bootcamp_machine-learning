import numpy as np
from TinyStatistician import TinyStatistician

def zscore_(x):
  tn = TinyStatistician()
  mean = tn.mean(x)
  std = tn.std(x)
  return (x - mean) / std

  
