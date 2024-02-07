import pandas as pd


def load_file(file_path, to_read):
    file = pd.read_csv(file_path, usecols=to_read)
    if 'generated' in to_read:
        file['generated_cor'] = file['generated'].replace({
            0: 'Human',
            1: 'AI'
        })
    else:
        file['generated_cor'] = file['source']
        file.loc[file['generated_cor'] != 'Human', 'generated_cor'] = 'AI'

    return file
