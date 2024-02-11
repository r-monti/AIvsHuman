import pandas as pd
import numpy as np


def load_file(file_path, to_read, balance=False):
    file = pd.read_csv(file_path, usecols=to_read)
    if 'generated' in to_read:
        file['generated_cor'] = file['generated'].replace({
            0: 'Human',
            1: 'AI'
        })
    else:
        file['generated_cor'] = file['source']
        file.loc[file['generated_cor'] != 'Human', 'generated_cor'] = 'AI'
    if balance:
        # Calculate counts of AI and Human samples
        ai_count = (file['generated_cor'] == 'AI').sum()
        human_count = (file['generated_cor'] == 'Human').sum()
        print('AI count:', ai_count)
        print('Human count:', human_count)
        # Balance the number of samples
        min_count = min(ai_count, human_count)
        if ai_count > min_count:
            print(f"I'll delete: {ai_count - min_count} from AI rows")
            ai_indices = np.random.choice(file[file['generated_cor'] == 'AI'].index, ai_count-min_count, replace=False)
            file = file.drop(ai_indices)
        elif human_count > min_count:
            print(f"I'll delete: {human_count - min_count} from Human rows")
            human_indices = np.random.choice(file[file['generated_cor'] == 'Human'].index, human_count-min_count, replace=False)
            file = file.drop(human_indices)

    return file
