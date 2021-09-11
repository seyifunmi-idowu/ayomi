from sklearn.model_selection import train_test_split
from sklearn import tree
import pandas as pd


# opening the data
# test_data = pd.read_csv('test.csv')
train_data = pd.read_csv('train.csv')

# removing the datas that are not necessary
train_labels = train_data['Survived']
train_data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
# print(train_data)

# mapping our data as our model can only read numerical values
train_data['Sex'] = train_data.Sex.map({"male":1,"female":0})
train_data['Embarked'] = train_data.Embarked.map({"C":0, "Q":1, "S":2})
# print(train_data)

# counting how many blank spaces we have
# for col in train_data.columns:
#     print(col, ":", train_data[col].isnull().sum())

# using median for the blank spaces
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].median())

# for col in train_data.columns:
#     print(col, ":", train_data[col].isnull().sum())

# train our model
x_train, x_test, y_train, y_test = train_test_split(train_data, train_labels, train_size=0.7)

decision_tree = tree.DecisionTreeClassifier(max_depth=3)
model = decision_tree.fit(x_train, y_train)

acc = model.score(x_test, y_test)
print(acc)


# going ahead to predicting our data
test_data = pd.read_csv('test.csv')
test_data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data['Sex'] = test_data.Sex.map({"male":1,"female":0})
test_data['Embarked'] = test_data.Embarked.map({"C":0, "Q":1, "S":2})
# test_data = pd.get_dummies(test_data)
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

# print(test_data)

pred = model.predict(x_test)
pred = pred.astype(int)
# submission = pd.read_csv('gender_submission.csv')
# submission['Survived'] = pred
# submission.to_csv('submission.csv', index=False)
