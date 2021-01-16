import pandas as pd
import numpy as np
from multivariate_linear_model import MyLinearRegression as MultiLR

data = pd.read_csv("../resources/spacecraft_data.csv")
Y = np.array(data[['Sell_price']])

# BY AGE
X_age = np.array(data[['Age']])
myLR_age = MultiLR([[1000.0], [-1.0]], alpha = 2.5e-5, max_iter = 100000)
myLR_age.fit_(X_age[:,0].reshape(-1,1), Y)
RMSE_age = myLR_age.mse_(X_age[:,0].reshape(-1,1), Y)
print("Age ok")

# BY THRUST POWER
X_power = np.array(data[['Thrust_power']])
myLR_power =  MultiLR([[0], [0]], alpha = 2.5e-5, max_iter = 100000)
myLR_power.fit_(X_power[:,0].reshape(-1,1), Y)
RMSE_power = myLR_power.mse_(X_power[:,0].reshape(-1,1), Y)
print("Thrust ok")

# BY TERAMETERS
X_tmetter = np.array(data[['Terameters']])
myLR_tera =  MultiLR([[0], [0]], alpha = 2.5e-5, max_iter = 100000)
myLR_tera.fit_(X_tmetter[:,0].reshape(-1,1), Y)
RMSE_power = myLR_tera.mse_(X_tmetter[:,0].reshape(-1,1), Y)
print("Terameters ok")


# PLOT
myLR_power.plot(X_power[:,0].reshape(-1,1), Y, x_label='x2: thrust power(in 10km/s)', y_label='y: sell price (in keuros)', label_predicted='Predicted sell price', label_normal='Sell price')
myLR_tera.plot(X_tmetter[:,0].reshape(-1,1), Y, x_label='distance totalizer value of spacecraft (in Tmeters)distance totalizer value of spacecraft (in Tmeters)', y_label='y: sell price (in keuros)', label_predicted='Predicted sell price', label_normal='Sell price')
myLR_age.plot(X_age, Y, x_label='x1: age (in years)', y_label='y: sell price (in keuros)', label_predicted='Predicted sell price', label_normal='Sell price')
