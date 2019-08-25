import numpy as np 

import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#taking care of messing data 
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
imp_mean = imp_mean.fit(X[:, 1:3])
X[:, 1:3] = imp_mean.transform(X[:, 1:3])
print('Taking care of missing data')
print (X[:])

#categorical data
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
labelencoder_X = LabelEncoder()
X[: , 0] = labelencoder_X.fit_transform(X[:, 0])
print ('enconding categorical variable')
print (X[:])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print ('after using dummy variable')
print (X)
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print ('Depending variable')
print (X)
#split the data set into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test , y_train, y_test = train_test_split(X,y , test_size = 0.2, random_state=0)
print (X)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler ()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print (X_test)
