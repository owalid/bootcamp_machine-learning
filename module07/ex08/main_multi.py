import pandas as pd
import numpy as np
from multivariate_linear_model import MyLinearRegression as MultiLR

data = pd.read_csv("../resources/spacecraft_data.csv")
Y = np.array(data[['Sell_price']])
X_age = np.array(data[['Age']])
X_power = np.array(data[['Thrust_power']])
X_tmetter = np.array(data[['Terameters']])

# MULTIVARIATE
X_multi = np.array(data[['Age','Thrust_power','Terameters']])
my_lreg = MultiLR([[1.0], [1.0], [1.0], [1.0]], alpha = 1e-6, max_iter = 600000)
my_lreg.fit_(X_multi, Y)
my_lreg.plot_multivariate(X_multi, X_age, Y, x_label='x1: age (in years)', y_label='y: sell price (in keuros)', label_predicted='Predicted sell price', label_normal='Sell price')
my_lreg.plot_multivariate(X_multi, X_power, Y, x_label='x2: thrust power(in 10km/s)', y_label='y: sell price (in keuros)', label_predicted='Predicted sell price', label_normal='Sell price')
my_lreg.plot_multivariate(X_multi, X_tmetter, Y, x_label='x3: distance totalizer value of  spacecraft (in Tmeters)', y_label='y: sell price (in keuros)', label_predicted='Predicted sell price', label_normal='Sell price')