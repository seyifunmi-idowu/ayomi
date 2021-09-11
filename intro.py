import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from sklearn.model_selection import train_test_split

#sep stands for seperate by
data = pd.read_csv("student-mat.csv", sep=";")

#we have cut the data to the the ones we need
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

'''#looking for the best model. so we are checking through 30 models to get the highest and use the highest
best = 0
for _ in range(30):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)
    # print ("X_train: \n", x_train)
    # print ("y_train: \n", y_train)
    # print("X_test: \n", x_test)
    # print ("y_test: \n", y_test)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    print(acc)

    if acc > best:
        best = acc
        #we are saving our model into this file. as we dont have to create new ones when we start another code
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)'''

#to read our pickle file
pickle_in = open("studentmodel.pickle", "rb")
#loading the pickle into our linear model
linear = pickle.load(pickle_in)


print(f"coefficient: {linear.coef_}\n")
print(f"intercept: {linear.intercept_}\n")

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(round(predictions[x],1), x_test[x], y_test[x])


# #plotting the graph
# p = "studytime"
# style.use("ggplot")
# #plotting the graph in a x and y axis
# pyplot.scatter(data[p], data['G3'])
# #naming the x axis
# pyplot.xlabel(p)
# #naming the y axis
# pyplot.ylabel("Final Grade")
# pyplot.show()





# X =  list(range(15))
# y = [x * x for x in X]
# print(X)
# print(y)
#
# """ x_train and x_test takes randomly from the x variable
# while y_train and y_test takes rancomly from the y variable"""
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.65,test_size=0.35)
# print ("X_train: ", X_train)
# print ("y_train: ", y_train)
# print("X_test: ", X_test)
# print ("y_test: ", y_test)