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
def countSamples(file, column="generated_cor"):
    plt.figure(figsize=(10, 6))

    # Get unique class names and colors
    class_names = file[column].unique()
    color_mapping = {'Human': '#00c851'}

    # Assign #ffbb33 color to non-'Human' values
    for value in file[column].unique():
        if value != 'Human':
            color_mapping[value] = '#ffbb33'

    # Create list of colors based on 'generated_cor' values
    colors = [color_mapping[value] for value in file[column].unique()]

    sns.countplot(x=column, data=file, palette=colors, hue=column, legend=False)

    plt.title('Quantity of samples by Author', fontsize=24)
    plt.ylabel("Number of samples")
    plt.xlabel('Author')
    plt.xticks(range(len(class_names)), class_names)

    plt.show()

def countSamples2(file):
    plt.figure(figsize=(10, 6))

    # Get unique class names and colors
    class_names = file['source'].unique()
    color_mapping = {'Human': '#00c851'}

    # Assign #ffbb33 color to non-'Human' values
    for value in file['source'].unique():
        if value != 'Human':
            color_mapping[value] = '#ffbb33'

    # Create list of colors based on 'generated_cor' values
    colors = [color_mapping[value] for value in file['source'].unique()]

    sns.countplot(x='source', data=file, palette=colors, hue='source', legend=False)

    plt.title('Quantity of samples by Author', fontsize=24)
    plt.ylabel("Number of samples")
    plt.xlabel('Author')
    plt.xticks(range(len(class_names)), class_names)

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

def main(file_path, to_read):
    file = pd.read_csv(file_path, usecols=to_read)
    if 'generated' in to_read:
        file['generated_cor'] = file['generated'].replace({
            0: 'Human',
            1: 'AI'
        })
    else:
        file['generated_cor'] = file['source']
        file.loc[file['generated_cor'] != 'Human', 'generated_cor'] = 'AI'

    print(file['generated_cor'].value_counts())
    # countSamples(file.copy())
    wordLength(file.copy())


if __name__ == "__main__":
    columns_to_read = ['text', 'generated']
    # main("/home/cristian/Downloads/archive/AI_Human.csv", columns_to_read)
    main("D:/Nicro/Downloads/AI_Human.csv", columns_to_read)

    columns_to_read2 = ['text', 'source']
    main("D:/Nicro/Downloads/archive/data.csv", columns_to_read2)
    # main("/home/cristian/Downloads/2/data.csv", columns_to_read)
