import numpy as np
import sys
import random
import pandas as pd
from torch.nn.functional import cosine_similarity, cross_entropy
from torch import Tensor, from_numpy
from tqdm import tqdm
from train import SimCLR
from prepare_data import get_embeddings

random_seed = 0
np.random.seed(random_seed)
random.seed(random_seed)

def load_eval_data():
    df = pd.read_csv('../dataset/df_file.csv')
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
        data.append(f'query: {text}')
        data.append(label)
        datas.append(data)
    dataset = pd.DataFrame(datas, columns=columns)
    return dataset

def evaluation(model:SimCLR):
    truth = []
    pred = []
    dataset = load_eval_data()
    for _, data in tqdm(dataset.iterrows(), total=len(dataset)):
        embeddings = from_numpy(get_embeddings(data.values[:-1], 'evaluation_embedding.pkl')).to('cuda')
        # embeddings = model(embeddings)
        similarity = [cosine_similarity(embed, embeddings[[-1]]).cpu().detach().numpy() for embed in embeddings[:-1]]
        truth.append(data['label'])
        pred.append(np.argmax(similarity))
    return cross_entropy(Tensor(pred), Tensor(truth)).item(), sum(Tensor(pred) == Tensor(truth)) / len(truth)

if __name__ == "__main__":
    model = SimCLR.load_from_checkpoint("lightning_logs/version_23/checkpoints/epoch=5-step=750.ckpt").to('cuda')
    model.eval()
    print(evaluation(model))