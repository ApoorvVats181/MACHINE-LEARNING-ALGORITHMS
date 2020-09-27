# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Salary_Data.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size =1 / 3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
print("Model Score : ", model.score(x_test,y_test)*100,'%')

# Predicting the Test set results
y_pred = model.predict(x)

# Visualising
plt.scatter(x_train, y_train, color ='blue', label = 'Training Data' )
plt.scatter(x_test, y_test, color ='green', label = 'Testing Data')

plt.plot(x, y_pred, color ='red', label = 'Best fit line')

plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()

plt.show()

