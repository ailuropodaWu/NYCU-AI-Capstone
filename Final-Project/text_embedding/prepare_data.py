import lightning as pl
import os
import numpy as np
import pandas as pd
import pickle
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
from openai import OpenAI
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
os.environ.get("OPENAI_API_KEY")

# Function to get embeddings from OpenAI and cache them
def get_embeddings(texts, cache_file='embeddings_cache.pkl'):
    client = OpenAI(organization="org-HYkhp068aByFBQ1iIDHMMvDY")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            embeddings_cache = pickle.load(f)
    else:
        embeddings_cache = {}

    embeddings = []
    new_embed = 0
    progress = tqdm(texts)
    for text in progress:
        if len(text) > 8192:
            text = text[:8192]
        if text in embeddings_cache:
            embeddings.append(embeddings_cache[text])
        else:
            response = client.embeddings.create(input=text, model="text-embedding-3-small").data
            embedding = response[0].embedding
            embeddings_cache[text] = embedding
            embeddings.append(embedding)
            new_embed += 1
    progress.close()
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings_cache, f)

    print(f'new request: {new_embed}')
    return np.array(embeddings, dtype='float32')

# Get embeddings for the texts
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx]

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, df, batch_size=32):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.embeddings = np.array(self.df['embedding'].values.tolist(), dtype='float32')
        self.setup()

    def setup(self, stage=None):
        self.train_dataset = EmbeddingDataset(self.embeddings[:int(len(self.embeddings) * 0.8)])
        self.val_dataset = EmbeddingDataset(self.embeddings[int(len(self.embeddings) * 0.8):])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)
       
def load_data():
    texts = load_dataset("wikipedia", "20220301.simple", split="train[:10000]")['text']
    dataset = pd.DataFrame()
    dataset['text'] = texts
    dataset['embedding'] = list(get_embeddings(texts))
    return dataset

    
if __name__ == "__main__":
    test_data_module = load_data()
    print(test_data_module)

