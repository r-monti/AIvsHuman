import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

from nltk import word_tokenize
from nltk.corpus import stopwords


def countSamplesGroup(file, column="group"):
    # Set plot size
    plt.figure(figsize=(10, 6))

    # Get unique class names and colors
    classNames = file[column].unique()
    colorMapping = dict()

    # Assign #ffbb33 color to non-'Human' values
    for label in classNames:
        if label != 'Human':
            colorMapping[label] = '#ffbb33'
        else:
            colorMapping[label] = '#00c851'

    # Create list of colors based on 'group' values
    colors = [colorMapping[value] for value in classNames]

    sns.countplot(x=column, data=file, palette=colors, hue=column, legend=False)

    # Add title and lables
    plt.title('Quantity of samples by Author', fontsize=24)
    plt.ylabel("Number of samples")
    plt.xlabel('Author')
    plt.xticks(range(len(classNames)), classNames)

    plt.show()


def countAllSamples(file, column="source"):
    if column not in file.columns:
        print("Column not in file!")
        return

    # Set plot size
    plt.figure(figsize=(10, 6))

    # Get unique class names and colors
    classNames = file[column].unique()
    colorMapping = dict()

    # Assign #ffbb33 color to non-'Human' values
    for label in classNames:
        if label != 'Human':
            colorMapping[label] = '#ffbb33'
        else:
            colorMapping[label] = '#00c851'

    # Create list of colors based on classNames values
    colors = [colorMapping[value] for value in classNames]

    sns.countplot(x=column, data=file, palette=colors, hue=column, legend=False)

    # Add title and lables
    plt.title('Quantity of samples by Author', fontsize=24)
    plt.ylabel("Number of samples")
    plt.xlabel('Author')
    plt.xticks(range(len(classNames)), classNames)

    # Rotate labels on the x axys
    plt.xticks(rotation=90)

    # Add margin to the bottom
    plt.subplots_adjust(bottom=0.4)

    # Make "Human" lable red
    ax = plt.gca()
    labels = ax.get_xticklabels()
    for i, label in enumerate(labels):
        if label.get_text() == 'Human':
            labels[i].set_color('red')

    # Show the plot
    plt.show()


def textLength(file):
    # Calcola le lunghezze dei testi e trasformale in una lista
    textLen = file['text'].apply(len).tolist()

    # Calcola la frequenza di ogni lunghezza del testo
    freqDict = {}
    for length in textLen:
        freqDict[length] = freqDict.get(length, 0) + 1

    # Ordina le lunghezze dei testi e le relative frequenze
    sortedLengths = sorted(freqDict.keys())
    freq = [freqDict[length] for length in sortedLengths]

    # Crea il grafico a linea della distribuzione delle lunghezze dei testi
    plt.figure(figsize=(10, 6))
    plt.plot(sortedLengths, freq, color='skyblue')
    plt.title('Distribution of Text Lengths')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.xscale('log')  # Scala logaritmica sull'asse x per una migliore visualizzazione
    plt.grid(True, which="both", ls="--")  # Aggiunge una griglia sia sulle x che sulle y

    plt.xticks(rotation=45)
    plt.show()


def countWords(file):
    # Concatena tutte le frasi in un'unica stringa
    all_text = ' '.join(file['text'][:10000])

    # Tokenizzazione delle parole
    tokens = word_tokenize(all_text)

    # Rimozione delle stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Calcola le frequenze delle parole
    word_freq = Counter(filtered_tokens)

    # Estrai le parole e le relative frequenze
    words = list(word_freq.keys())  # Estrai le parole
    frequencies = list(word_freq.values())  # Estrai le frequenze

    # Ordina le parole in base alle frequenze
    words, frequencies = zip(*sorted(zip(words, frequencies), key=lambda x: x[1], reverse=True))

    # Seleziona le 100 parole più comuni
    top_words = list(zip(words[:100], frequencies[:100]))

    # Estrai le parole e le relative frequenze
    words, frequencies = zip(*top_words)

    # Crea il grafico
    plt.figure(figsize=(15, 8))
    plt.bar(words, frequencies)
    plt.xlabel('Parole')
    plt.ylabel('Numero di occorrenze')
    plt.title('Top 100 Parole più Usate')
    plt.xticks(rotation=90)  # Ruota le etichette sull'asse x per una migliore leggibilità
    plt.show()
    """
    # Concatena tutte le frasi in un'unica stringa
    all_text = ' '.join(file['text'][:100000])

    # Tokenizzazione delle parole
    tokens = word_tokenize(all_text)

    # Rimozione delle stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

    # Calcola le frequenze delle parole
    word_freq = Counter(filtered_tokens)

    # Seleziona le 100 parole più comuni
    top_words = word_freq.most_common(100)

    # Estrai le parole e le relative frequenze
    words, frequencies = zip(*top_words)

    # Crea il grafico
    plt.figure(figsize=(15, 8))
    plt.bar(words, frequencies)
    plt.xlabel('Parole')
    plt.ylabel('Frequenza')
    plt.title('Top 100 Parole più Usate')
    plt.xticks(rotation=90)  # Ruota le etichette sull'asse x per una migliore leggibilità
    plt.show()
    """
    """
    # Combine all text data into a single string
    allText = ' '.join(file['text'][:100000])

    # Tokenization
    tokens = wordTokenize(allText)

    # Count word frequencies
    wordFreq = Counter(tokens)

    # Plotting the most frequent words
    mcWords = wordFreq.most_common(100)

    words, frequencies = zip(*mcWords)

    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.title('Top 100 Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.show()
    """


def prepareFile(filePath, toRead):
    file = pd.read_csv(filePath, usecols=toRead)

    if 'generated' in toRead:
        file['group'] = file['generated'].replace({
            0: 'Human',
            1: 'AI'
        })
    else:
        file['group'] = file['source']
        file.loc[file['group'] != 'Human', 'group'] = 'AI'

    return file


def startML(filePath, toRead):
    file = prepareFile(filePath, toRead)

    print(file['group'].value_counts())
    # countSamplesGroup(file.copy())
    # countAllSamples(file.copy())
    # textLength(file.copy())
    countWords(file.copy())


if __name__ == "__main__":
    columns_to_read = ['text', 'generated']
    columns_to_read2 = ['text', 'source']

    startML("C:/AI_Human.csv", columns_to_read)
    # startML("/home/cristian/Downloads/archive/AI_Human.csv", columns_to_read)

    startML("C:/data.csv", columns_to_read2)
    # startML("/home/cristian/Downloads/2/data.csv", columns_to_read)
