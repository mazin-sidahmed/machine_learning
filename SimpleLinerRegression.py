# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
print (X_train)
print("=====================")
print(X_test)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""
#fitting simple linear regression to the training data
from sklearn.linear_model import LinearRegression
regrosser = LinearRegression ()
regrosser.fit(X_train,y_train)
print ("xxxxxxxxxxxxxx")
print (X_train)
print("yyyyyyyyyyyyyyyyy")
print (y_train)
# predicting the test set result
y_predict = regrosser.predict(X_test)
print ("y the real one ")
print(y_test)
print ("the predicted y")
print (y_predict)
# visulizing the traing set result
plt.scatter(X_train , y_train , color = 'red')
plt.plot(X_train , regrosser.predict(X_train) )
plt.title('Salary vs Exprience (Training set) ')
plt.xlabel ('Years of Experience')
plt.ylabel ('Salary')
plt.show () 
# visulizing the test set result
plt.scatter(X_test , y_test , color = 'red')
plt.plot(X_train , regrosser.predict(X_train) )
plt.title('Salary vs Exprience (Test set) ')
plt.xlabel ('Years of Experience')
plt.ylabel ('Salary')
plt.show () 