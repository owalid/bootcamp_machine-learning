import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR

data = pd.read_csv("../resources/are_blue_pills_magics.csv")
Xpill = np.array(data["Micrograms"]).reshape(-1,1)
Yscore = np.array(data["Score"]).reshape(-1,1)

print(Xpill)
print(Yscore)

linear_model1 = MyLR(np.array([[89.0], [-8]]))
linear_model2 = MyLR(np.array([[89.0], [-6]]))

# linear_model1.plot(Xpill, Yscore)
# linear_model2.plot(Xpill, Yscore)

Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)

print("mine: ", linear_model1.mse_(Yscore, Y_model1)) #MY
# 57.60304285714282

print("not mine: ", mean_squared_error(Yscore, Y_model1))
# 57.603042857142825

print("mine: ", linear_model2.mse_(Yscore, Y_model2)) # MY
# 232.16344285714285

print("not mine: ", mean_squared_error(Yscore, Y_model2))
# 232.16344285714285

# linear_model1.plot(Xpill, Y_model1)
# linear_model2.plot(Xpill, Y_model2)