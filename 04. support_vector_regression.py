# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Position_Salaries.csv')
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values
print(x)
print(y)
y = y.reshape(len(y),1)
print(y)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
x = sc_X.fit_transform(x)
y = sc_y.fit_transform(y)
print(x)
print(y)

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
model = SVR(kernel ='rbf')
model.fit(x, y)
print('Accuracy of model1 : ',model.score(x,y)*100,'%')
# Predicting a new result
sc_y.inverse_transform(model.predict(sc_X.transform([[6.5]])))

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(x), sc_y.inverse_transform(y), color ='red')
plt.plot(sc_X.inverse_transform(x), sc_y.inverse_transform(model.predict(x)), color ='blue')
plt.title('Position Vs Salary (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(x)), max(sc_X.inverse_transform(x)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(x), sc_y.inverse_transform(y), color ='red')
plt.plot(X_grid, sc_y.inverse_transform(model.predict(sc_X.transform(X_grid))), color ='blue')
plt.title('Position Vs Salary 2 (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()