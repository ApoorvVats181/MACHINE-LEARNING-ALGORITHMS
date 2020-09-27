# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv('Restaurant_Reviews.tsv', delimiter ='\t', quoting = 3)     # quoting = 3 ===>>ignores double quotes

# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])  # ^ only keeps A,a to Z,z # second parameter replaces removed elements with space
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]   # stopwords contains useless words
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)

# Predicting the Test set results
y_pred = model.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import accuracy_score,confusion_matrix
print('Accuracy score : ',accuracy_score(y_test,y_pred)*100,'%')
print('Confusion Matrix : \n',confusion_matrix(y_test, y_pred))




