import gc

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed


def run_pipeline(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred


def evaluate_model(model, y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


def main(file_path, to_read):
    file = pd.read_csv(file_path, usecols=to_read)

    X = file['text']
    if 'generated' in to_read:
        file['generated_cor'] = file['generated'].replace({
            1: 'Human',
            0: 'AI'
        })
    else:
        file['generated_cor'] = file['source']
        file.loc[file['generated_cor'] != 'Human', 'generated_cor'] = 'IA'
    y = file['generated_cor']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    pipelines = [
        Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('clf', MultinomialNB())]),
        Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('clf', ComplementNB())]),
        Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('clf', DecisionTreeClassifier(max_depth=5))])
    ]

    results = Parallel(n_jobs=-1)(delayed(run_pipeline)(pipe, X_train, y_train, X_test) for pipe in pipelines)

    accuracies = []
    reports = []

    for i, result in enumerate(results):
        accuracy, report = evaluate_model(pipelines[i], y_test, result)
        accuracies.append(accuracy)
        reports.append(report)

        del result
        gc.collect() #test

    model_names = ["Multinomial Naive Bayes", "Complement Naive Bayes", "Decision Tree Classifier"]
    for i in range(len(model_names)):
        print(f"{model_names[i]}:")
        print(f"Accuracy: {accuracies[i] * 100:.2f}%")
        print("Classification Report:")
        print(reports[i])
        print()


if __name__ == "__main__":
    columns_to_read = ['text', 'generated']
    #main("/home/cristian/Downloads/archive/AI_Human.csv", columns_to_read)
    main("D:/Nicro/Downloads/AI_Human.csv", columns_to_read)

    #columns_to_read = ['text', 'source']
    #main("D:/Nicro/Downloads/archive/data.csv", columns_to_read)
    #main("/home/cristian/Downloads/2/data.csv", columns_to_read)
