from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


data = pd.read_csv("car.data")

#we are trying to change the datas in the car.data to int values so as to make working easy
le = LabelEncoder()
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

x_train, x_test, y_train, y_test =train_test_split(x, y, test_size=0.1)

# model = KNeighborsClassifier(n_neighbors=9)

LogisticRegressionPipeline = Pipeline([('scaler1', MinMaxScaler()),
                                       ('mypca', PCA(n_components=3)),
                                       ('logistics_classifier', LogisticRegression())])

DecisionTreePipeline = Pipeline([('scaler2', MinMaxScaler()),
                                 ('mypca', PCA(n_components=3)),
                                 ('decisiontree_classifier', DecisionTreeClassifier())])

KNeighborsPipeline = Pipeline([('scaler3', MinMaxScaler()),
                               ('mypca', PCA(n_components=3)),
                               ('kneighbors_classifier', KNeighborsClassifier(n_neighbors=5))])



mypipeline = [LogisticRegressionPipeline, DecisionTreePipeline, KNeighborsPipeline]

accuracy = 0.0
classifier = 0
pipeline = ''

pipe_dict = {0 : 'Logistic Regression',
             1 : 'Decision Tree',
             2 : 'K Neighbours'}

# model.fit(x_train, y_train)

for pipe in mypipeline:
    pipe.fit(x_train, y_train)

for x, model in enumerate(mypipeline):
    print(f"{pipe_dict[x]} Test Accuracy {model.score(x_test, y_test)}")

# acc = model.score(x_test, y_test)
# print(acc)

for x, model in enumerate(mypipeline):
    if model.score(x_test, y_test) > accuracy:
        accuracy = model.score(x_test, y_test)
        pipeline = model
        classifier = x

    print(f"classifier with best accuracy: {pipe_dict[classifier]}")



predict = pipeline.predict(x_test)
name = ["unacc", "acc", "good", "vgood"]

for x in range(len(predict)):
    print(f"predicted: {name[predict[x]]}, Data {x_test[x]}, Actual: {name[y_test[x]]}")

