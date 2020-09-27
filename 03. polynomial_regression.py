# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Position_Salaries.csv')
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(x, y)

pred_y = model1.predict(x)
plt.plot(x, pred_y, color='b', label='Predicted by model1')
print('Accuracy of model1 : ',model1.score(x,y)*100,'%')



# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
x_poly = poly.fit_transform(x)
model2 = LinearRegression()
model2.fit(x_poly, y)

pred_y = model2.predict(x_poly)
plt.plot(x, pred_y, color='g',label='Predicted by model2')
print('Accuracy of model2 : ',model2.score(x_poly,y)*100,'%')



# Visualising
plt.scatter(x, y, color ='red')
plt.title('Position Vs Salaries')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()





# Predicting a new result with Linear Regression
model1.predict([[6.5]])

# Predicting a new result with Polynomial Regression
model2.predict(poly.fit_transform([[6.5]]))