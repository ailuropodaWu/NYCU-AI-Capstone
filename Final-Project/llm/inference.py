import os
import pandas as pd
import json
import lightning as L
from model import LLMEmbeddingModel
from tqdm import tqdm

def get_embedding(save_root, ckpt, use_saved=True, data_type="paragraph"):
    assert data_type in ['sentence', 'paragraph'], "invalid data type"
    save_path = os.path.join(save_root, f"{data_type}_embedding_dict.json")
    dataset_path = f"./dataset/{data_type}_data.json"
    df = pd.read_json(dataset_path)
    print(f"Getting {data_type} embedding ...")
    
    if use_saved:
        with open(save_path, "r") as fp:
            embedding_dict = json.load(fp)
        return embedding_dict
    
    model = LLMEmbeddingModel.load_from_checkpoint(model)
    
    embedding_dict = {}
    datas = set()
    batch = []
    batch_cnt = 0
    batch_size = 1
    for _, row in tqdm(df.iterrows()):
        datas.add(row[f"{data_type}1"])
        datas.add(row[f"{data_type}2"])
        

    for data in tqdm(datas):
        embedding_dict[data] = batch_cnt
        batch.append(data)
        batch_cnt += 1
        if batch_cnt == batch_size:
            batch_cnt = 0
            embeddings = model.forward(batch)
            for s in batch:
                embedding_dict[s] = embeddings[embedding_dict[s]]
            batch = []
    if len(batch) > 0:
        embeddings = model.forward(batch)
        for s in batch:
            embedding_dict[s] = embeddings[embedding_dict[s]]
    with open(save_path, "w") as fp:
        json.dump(embedding_dict, fp)
    return embedding_dict
    