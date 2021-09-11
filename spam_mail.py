from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import pickle

data = pd.read_csv('spam.csv', encoding="latin-1")

data.head(5)

data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

data['v1']=data['v1'].map({'ham' : 0, 'spam' : 1})

cv = CountVectorizer()

x = data['v2']
y = data['v1']

x = cv.fit_transform(x)
# print(x)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

model = MultinomialNB()

model.fit(x_train, y_train)

acc = model.score(x_test,  y_test)
print(acc)

# pickle.dump(model, open('spam.pkl', "wb"))
# pickle.dump(cv, open('vectorizerr.pkl', "wb"))

clf = pickle.load(open("spam.pkl","rb"))

msg = "Your credits have been topped up for http://www.bubbletext.com Your renewal Pin is tgxxrz"
data = [msg]
vect = cv.transform(data).toarray()
result = model.predict(vect)
print(result)