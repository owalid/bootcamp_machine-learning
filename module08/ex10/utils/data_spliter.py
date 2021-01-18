import numpy as np

def data_spliter(x, y, proportion):
  np.random.shuffle(x)
  np.random.shuffle(y)
  x_train_len = int(x.shape[0] * proportion)
  y_train_len = int(y.shape[0] * proportion)
  return (x[:x_train_len], x[x_train_len:x.shape[0]], y[:y_train_len], y[y_train_len:y.shape[0]])
