import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

'''
def countSamples(file, column):
    plt.figure(figsize=(10, 6))
    class_names = file['generated_cor'].unique()
    color_mapping = {'Human': '#00c851', 'AI': '#ffbb33'}
    colors = [color_mapping[value] for value in file['generated_cor'].unique()]

    sns.countplot(x='generated_cor', data=file, palette=colors, hue='generated_cor', legend=False)

    plt.title('Quantity of samples by Author', fontsize=24)
    plt.ylabel("Number of samples")
    plt.xlabel('Author')
    plt.xticks(range(len(class_names)), class_names)

    plt.show()
'''


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

    # Create list of colors based on 'generated_cor' values
    colors = [colorMapping[value] for value in classNames]

    sns.countplot(x=column, data=file, palette=colors, hue=column, legend=False)

    plt.title('Quantity of samples by Author', fontsize=24)
    plt.ylabel("Number of samples")
    plt.xlabel('Author')
    plt.xticks(range(len(classNames)), classNames)

    plt.show()


def countAllSamples(file, column="source"):
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


def wordLength(file):
    file['text_length'] = file['text'].apply(len)

    file['text_word_count'] = file['text'].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(16, 8))

    # First subplot for text length distribution
    plt.subplot(1, 2, 1)
    n, bins, patches = plt.hist(file['text_length'], bins=10, color='mediumaquamarine', edgecolor='black',
                                alpha=0.5, label='News Article')
    plt.grid(linestyle='--', alpha=0.6)
    plt.xlabel("Text Length", fontsize=10, color='black')
    plt.ylabel("Frequency", fontsize=10, color='black')
    plt.title(f'Text Length Distribution', fontsize=12, color='black')

    # Annotate the plot with bin values (vertical text)
    for bin_val, freq in zip(bins, n):
        plt.text(bin_val + 100, freq + 20, f'{int(freq)}', ha='left', va='baseline', fontsize=12)

    # Second subplot for word count distribution
    plt.subplot(1, 2, 2)
    n, bins, patches = plt.hist(file['text_word_count'], bins=10, color='violet', edgecolor='black', alpha=0.7,
                                label='News Article')
    plt.grid(linestyle='--', alpha=0.6)
    plt.xlabel("Word Count", fontsize=10, color='black')
    plt.ylabel("Frequency", fontsize=10, color='black')
    plt.title(f'Word Count Distribution', fontsize=12, color='black')

    # Annotate the plot with bin values (vertical text)
    for bin_val, freq in zip(bins, n):
        plt.text(bin_val + 20, freq + 20, f'{int(freq)}', ha='left', va='baseline', fontsize=12)

    # Adjust the layout for subplots
    plt.tight_layout()

    # Show the plot
    plt.show()


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
    countSamplesGroup(file.copy())
    countAllSamples(file.copy())
    # wordLength(file.copy())


if __name__ == "__main__":
    # columns_to_read = ['text', 'generated']
    columns_to_read2 = ['text', 'source']

    # startML("C:/AI_Human.csv", columns_to_read)
    # startML("/home/cristian/Downloads/archive/AI_Human.csv", columns_to_read)

    startML("C:/data.csv", columns_to_read2)
    # startML("/home/cristian/Downloads/2/data.csv", columns_to_read)
