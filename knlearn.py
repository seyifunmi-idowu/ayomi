import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing


data = pd.read_csv("car.data")

#we are trying to change the datas in the car.data to int values so as to make working easy
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
log_boot = le.fit_transform(list(data["log_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

#zip() creats tuple objects with all of the differnt values passed into the list
x = list(zip(buying, maint, door, persons, log_boot, safety))
y = list(cls)
# for a in x:
#     print(a)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predict = model.predict(x_test)
name = ["unacc", "acc", "good", "vgood"]

# for x in range(len(predict)):
#     print(f"predicted: {name[predict[x]]}, Data {x_test[x]}, Actual: {name[y_test[x]]}")

