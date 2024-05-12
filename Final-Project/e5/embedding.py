import torch
import torch.nn.functional as F
import pandas as pd
import json

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

data_path = "./dataset/data.json"
df = pd.read_json(data_path)
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
model = AutoModel.from_pretrained('intfloat/e5-large-v2')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


# Each input text should start with "query: " or "passage: ".
# For tasks other than retrieval, you can simply use the "query: " prefix.
def get_embedding(save_path="embedding_dict.json", use_saved=True):
    embedding_dict = {}
    sentences = set()
    batch = []
    batch_size = 32
    batch_cnt = 0
    print("Getting Embedding ...")
    
    if use_saved:
        with open(save_path, "r") as fp:
            embedding_dict = json.load(fp)
        return embedding_dict
    for _, row in tqdm(df.iterrows()):
        sentences.add(row["sentence1"])
        sentences.add(row["sentence2"])
    for sentence in tqdm(sentences):
        embedding_dict[sentence] = batch_cnt
        batch.append(sentence)
        batch_cnt += 1
        if batch_cnt == batch_size:
            batch_cnt = 0
            batch_dict = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = model(**batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            for s in batch:
                embedding_dict[s] = embeddings[embedding_dict[s]].cpu().detach().tolist()
            batch = []
            
    if len(batch) > 0:
        batch_dict = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        for s in batch:
            embedding_dict[s] = embeddings[embedding_dict[s]].cpu().detach().tolist()
        
    with open(save_path, "w") as fp:
        json.dump(embedding_dict, fp)
    return embedding_dict
