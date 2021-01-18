import pandas as pd
import numpy as np
from utils.data_spliter import data_spliter
from utils.zscore import zscore_
from utils.my_logistic_regression import MyLogisticRegression as MyLR

def get_category(y, category):
  cpy_y = y.copy()
  is_in_arr = y != category
  is_not = y == category
  cpy_y[is_in_arr] = 0
  cpy_y[is_not] = 1
  return cpy_y[:, 4]


data_planets_numbers = pd.read_csv('../resources/solar_system_census_planets.csv')
data_planets_numbers.head()
data_planets = pd.read_csv('../resources/solar_system_census.csv')
data_planets.head()

x = zscore_(np.array(data_planets_numbers['Origin']))
y = np.array(pd.concat([data_planets, data_planets_numbers['Origin']], axis=1))

x_train, x_test, y_train, y_test = data_spliter(x, y, 0.4)

y_0 = get_category(y, 0)
y_1 = get_category(y, 1)
y_2 = get_category(y, 2)
y_3 = get_category(y, 3)

print(y_1)
print(x)
# print(np.array([1, 1, 1, 1, 1]))

my_linear_0 = MyLR(np.array([1, 1, 1, 1, 1])
# my_linear_1 = MyLR([[1], [1], [1], [1], [1]])
# my_linear_2 = MyLR([[1], [1], [1], [1], [1]])
# my_linear_3 = MyLR([[1], [1], [1], [1], [1]])
