import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.impute import SimpleImputer
#importing the data
data = pd.read_csv("train.csv")

#filling the missing data
median = data["Age"].median()
data["Age"].fillna(median, inplace=True)

#choosing numerical data to work with
data = data[["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]]

predict = "Survived"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#training the model
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

linear = linear_model.LinearRegression()

#scoring and fitting the model
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

print(accuracy)

#opening pickle file
# with open("titanicmodel.pickle", "rb") as f:
# 	pickle.dump(linear,f)
#
# #saving model in pickle file
# pickle_in = open("titanicmodel", "rb")
# linear = pickle.load(pickle_in)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
	print(predictions[x], x_test[x], y_test[x])

