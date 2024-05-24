import torch
import torch.nn.functional as F
import pandas as pd
import json
import os
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


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
def get_embedding(save_root, use_saved=True, data_type='paragraph'):
    assert data_type in ['sentence', 'paragraph'], "invalid data type"
    save_path = os.path.join(save_root, f"{data_type}_embedding_dict.json")
    data_path = f"./dataset/{data_type}_data.json"
    df = pd.read_json(data_path)
    embedding_dict = {}
    datas = set()
    batch = []
    batch_size = 32 if data_type == 'sentence' else 4
    batch_cnt = 0
    print(f"Getting {data_type} embedding ...")
    
    if use_saved:
        with open(save_path, "r") as fp:
            embedding_dict = json.load(fp)
        return embedding_dict
    for _, row in tqdm(df.iterrows()):
        datas.add(row[f"{data_type}1"])
        datas.add(row[f"{data_type}2"])
    for data in tqdm(datas):
        embedding_dict[data] = batch_cnt
        batch.append(data)
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
