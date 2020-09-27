# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
import statsmodels

df = pd.read_csv('50_Startups.csv')
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print(x)

# Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 7)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
print("Model Score : ", model.score(x_test,y_test)*100,'%')

# Predicting the Test set results
y_pred = model.predict(x)
