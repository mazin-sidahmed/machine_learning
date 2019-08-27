# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[: , 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print (X_train)
print("=====================")
print(X_test)

# fitting multiple linear regression to the training set 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train , y_train)

# predicting the test set results
y_predict = regressor.predict(X_test)

# bulding the optimal model using backword elimination
import statsmodels.formula.api as smf
X = np.append(arr = np.ones((50,1)).astype(int) , values = X , axis=1)
optimal_X = X[:,[0, 1, 2, 3, 4, 5]] 
regressor_OrdinaryLeastSquere =smf.ols(endog = y ,exog = optimal_X ).fit()
regressor_OrdinaryLeastSquere.summary()