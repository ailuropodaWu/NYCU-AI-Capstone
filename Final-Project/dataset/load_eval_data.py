import pandas as pd
import numpy as np
import random

random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)

def load_eval_data():
    df = pd.read_csv('df_file.csv')
    splited_df = [group['Text'].values for _, group in df.groupby('Label')]
    data_size = 1000
    datas = []
    columns = ['sentence1', 'sentence2', 'sentence3', 'sentence4', 'sentence5', 'sample', 'label']
    for _ in range(data_size):
        data = []
        sample = df.iloc[random.randint(0, len(df) - 1)]
        text, label = sample['Text'], sample['Label']
        for cls in range(len(splited_df)):
            data.append(splited_df[cls][random.randint(0, len(splited_df[cls]) - 1)])
        data.append(text)
        data.append(label)
        datas.append(data)
    dataset = pd.DataFrame(datas, columns=columns)
    return dataset
        

if __name__ == "__main__":
    load_eval_data()