import sklearn
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

boston = load_boston()

x = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target)

#print(dataframe_x.head())

reg = linear_model.LinearRegression()

#splitting the data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#training model with our tests set
reg.fit(x_train, y_train)

#test data predictons
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

#check model performance using mean square error
#print(np.mean((y_pred - y_test)**2))
#check model performance using sklearn
#print( mean_squared_error(y_test, y_pred) )