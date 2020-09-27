# Support Vector Machine (SVM)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Social_Network_Ads.csv')
x = df.iloc[:, [2, 3]].values
y = df.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Training the SVM model on the Training set
from sklearn.svm import SVC
model = SVC(kernel ='linear', random_state = 0)
model.fit(x_train, y_train)

# Predicting the Test set results
y_pred = model.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import accuracy_score,confusion_matrix
print('Accuracy score : ',accuracy_score(y_test,y_pred)*100,'%')
print('Confusion Matrix : \n',confusion_matrix(y_test, y_pred))

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X1, X2 = np.meshgrid(np.arange(start =x_train[:, 0].min() - 1, stop =x_train[:, 0].max() + 1, step = 0.01),
                     np.arange(start =x_train[:, 1].min() - 1, stop =x_train[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_train)):
    plt.scatter(x_train[y_train == j, 0], x_train[y_train == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X1, X2 = np.meshgrid(np.arange(start =x_test[:, 0].min() - 1, stop =x_test[:, 0].max() + 1, step = 0.01),
                     np.arange(start =x_test[:, 1].min() - 1, stop =x_test[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_test)):
    plt.scatter(x_test[y_test == j, 0], x_test[y_test == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()