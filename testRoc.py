import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

columns_to_read = ['text', 'generated']

file = pd.read_csv("C:/AI_Human.csv", usecols=columns_to_read)

file['group'] = file['generated']
file['group'] = file['generated'].replace({
            0: 'Human',
            1: 'AI'
        })

pipeMNB = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('clf', MultinomialNB())])

x_train, x_test, y_train, y_test = train_test_split(file['text'], file['group'], test_size=0.2)

pipeMNB.fit(x_train, y_train)

# Evaluate accuracy and print classification report
predictMNB = pipeMNB.predict(x_test)
accuracy = accuracy_score(y_test, predictMNB)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report for Multinomial Naive Bayes:")
print(classification_report(y_test, predictMNB))


# Predict probabilities for the positive class
probs = pipeMNB.predict_proba(x_test)[:, 0]

# Convert categorical labels to binary labels
#y_test_binary = y_test.map({'Human': 0, 'AI': 1})
y_test_binary = (y_test == 'AI').astype(int)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test_binary, probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion matrix
conf_matrix = confusion_matrix(y_test, predictMNB)

# Plotting the confusion matrix
plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['AI', 'Human']
plt.xticks([0, 1], classes, rotation=45)
plt.yticks([0, 1], classes)

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.subplots_adjust(bottom=0.2)

for i in range(2):
    for j in range(2):
        plt.text(j, i, conf_matrix[i, j], horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")

plt.show()
