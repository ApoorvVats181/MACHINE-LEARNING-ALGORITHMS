# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Position_Salaries.csv')
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Training the Decision Tree Regression model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 0)
model.fit(x, y)

# Predicting a new result
print('Prediction : ',model.predict([[6.5]]))

# Visualising the Decision Tree Regression results
# This visualization is not correct because in this case it is non-continuous model
plt.scatter(x, y, color ='red')
plt.plot(x, model.predict(x), color ='blue')
plt.title('Position Vs Salary (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Decision Tree Regression results (higher resolution)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color ='red')
plt.plot(x_grid, model.predict(x_grid), color ='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()