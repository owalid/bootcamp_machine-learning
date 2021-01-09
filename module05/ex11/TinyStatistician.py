import math

class TinyStatistician:
  def mean(self, x):
    len_x = len(x)
    result = 0
    for i in range(len_x):
      result += float(x[i])
    return float(result / len_x)

  def median(self, x):
    x.sort()
    middle = int(len(x) / 2)
    return float(x[middle])

  def quartile(self, x, percentile):
    x.sort()
    if percentile == 25:
      first = int(len(x) * 0.25)
      return float(x[first])
    else:
      second = int(len(x) * 0.75)
      return float(x[second])

  def var(self, x):
    mean = self.mean(x)
    len_x = len(x)
    result = 0
    for i in range(len_x):
      result += (x[i] - mean)**2
    return float(result / len_x)

  def std(self, x):
    return math.sqrt(self.var(x))
