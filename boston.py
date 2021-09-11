from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


boston = load_boston()

x = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# reg = LinearRegression()

LinearRegressionPipeline = Pipeline([('scaler1', MinMaxScaler()),
                                       ('mypca', PCA(n_components=3)),
                                       ('linear_regression ', LinearRegression())])

DecisionTreePipeline = Pipeline([('scaler2', MinMaxScaler()),
                                 ('mypca', PCA(n_components=3)),
                                 ('decisiontree_classifier', DecisionTreeClassifier())])

KNeighborsPipeline = Pipeline([('scaler3', MinMaxScaler()),
                               ('mypca', PCA(n_components=3)),
                               ('kneighbors_classifier', KNeighborsClassifier(n_neighbors=5))])

mypipeline = [LinearRegressionPipeline, DecisionTreePipeline, KNeighborsPipeline]

accuracy = 0.0
classifier = 0
pipeline = ''

pipe_dict = {0 : 'Linear Regression',
             1 : 'Decision Tree',
             2 : 'K Neighbours'}

# reg.fit(x_train, y_train)
for pipe in mypipeline:
    pipe.fit(x_train, y_train)

for x, model in enumerate(mypipeline):
    if model.score(x_test, y_test) > accuracy:
        accuracy = model.score(x_test, y_test)
        pipeline = model
        classifier = x

#test data predictons
y_pred = pipeline.predict(x_test)
print(y_pred)
print(y_test)

#check model performance using mean square error
#print(np.mean((y_pred - y_test)**2))
#check model performance using sklearn
#print( mean_squared_error(y_test, y_pred) )