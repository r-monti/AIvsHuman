import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

columns_to_read = ['text', 'source']

file = pd.read_csv("/home/cristian/Downloads/2/data.csv", usecols=columns_to_read)

file['source'].value_counts()

file['generated_cor'] = file['source']
file.loc[file['generated_cor'] != 'Human', 'generated_cor'] = 'IA'

file['generated_cor'].value_counts()

pipeMNB = Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),('clf',MultinomialNB())])
pipeCNB = Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),('clf',ComplementNB())])
pipeDTC = Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),('clf',DecisionTreeClassifier(max_depth=5))])

x_train,x_test,y_train,y_test=train_test_split(file['text'], file['generated_cor'],test_size=0.2)

pipeMNB.fit(x_train,y_train)
pipeCNB.fit(x_train,y_train)
pipeDTC.fit(x_train,y_train)

predictMNB = pipeMNB.predict(x_test)
predictCNB = pipeCNB.predict(x_test)
predictDTC = pipeDTC.predict(x_test)

mnb = accuracy_score(y_test,predictMNB)
cnb = accuracy_score(y_test,predictCNB)
dtc = accuracy_score(y_test,predictDTC)

print(f"MNB: {mnb*100:.2f}%")
print(f"CNB: {cnb*100:.2f}%")
print(f"DTC: {dtc*100:.2f}%")

print("Classification Report for Multinomial Naive Bayes:")
print(classification_report(y_test, predictMNB))

print("\nClassification Report for Complement Naive Bayes:")
print(classification_report(y_test, predictCNB))

print("\nClassification Report for Decision Tree Classifier:")
print(classification_report(y_test, predictDTC))