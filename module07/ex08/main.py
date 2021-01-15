import pandas as pd
import numpy as np
from mylinearregression import MyLinearRegression as MyLR


data = pd.read_csv("spacecraft_data.csv")
X = np.array(data[['Age']])
Y = np.array(data[['Sell_price']])
myLR_age = MyLR([[1000.0], [-1.0]])
myLR_age.fit_(X[:,0].reshape(-1,1), Y, alpha = 2.5e-5, n_cycle = 100000)

RMSE_age = myLR_age.mse_(X[:,0].reshape(-1,1),Y)
print(RMSE_age)
# 57636.77729...





# data = pd.read_csv("spacecraft_data.csv")
# X = np.array(data[['Age','Thrust_power','Terameters']])
# Y = np.array(data[['Sell_price']])
# my_lreg = MyLR([1.0, 1.0, 1.0, 1.0])

# print(my_lreg.mse_(X,Y))
# # Output:
# # 144044.877...

# my_lreg.fit_(X,Y, alpha = 1e-4, n_cycle = 600000)
# print(my_lreg.theta)
# # Output:
# # array([[334.994...],[-22.535...],[5.857...],[-2.586...]])

# print(my_lreg.mse_(X,Y))
# # Output:
# # 586.896999...