import pandas as pd
import time
from tqdm import tqdm
import json
import os
from openai import OpenAI
from argparse import ArgumentParser
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())
os.environ.get("OPENAI_API_KEY")


def get_embedding(save_root, use_saved=True, data_type="paragraph"):
    assert data_type in ['sentence', 'paragraph'], "invalid data type"
    save_path = os.path.join(save_root, f"{data_type}_embedding_dict.json")
    dataset_path = f"./dataset/{data_type}_data.json"
    df = pd.read_json(dataset_path)
    print(f"Getting {data_type} embedding ...")
    
    if use_saved:
        with open(save_path, "r") as fp:
            embedding_dict = json.load(fp)
        return embedding_dict
    api_key = None
    client = OpenAI(organization="org-HYkhp068aByFBQ1iIDHMMvDY")
    
    
    embedding_dict = {}
    datas = set()
    batch = []
    batch_cnt = 0
    batch_size = 128
    for _, row in tqdm(df.iterrows()):
        datas.add(row[f"{data_type}1"])
        datas.add(row[f"{data_type}2"])

    for data in tqdm(datas):
        embedding_dict[data] = batch_cnt
        batch.append(data)
        batch_cnt += 1
        if batch_cnt == batch_size:
            batch_cnt = 0
            embeddings = client.embeddings.create(input=batch, model="text-embedding-3-small").data
            for s in batch:
                embedding_dict[s] = embeddings[embedding_dict[s]].embedding
            batch = []
    if len(batch) > 0:
        embeddings = client.embeddings.create(input=batch, model="text-embedding-3-small").data
        for s in batch:
            embedding_dict[s] = embeddings[embedding_dict[s]].embedding
    with open(save_path, "w") as fp:
        json.dump(embedding_dict, fp)
    return embedding_dict

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--not_use_saved", action="store_true", default=False)
    args = parser.parse_args()
    use_saved = not args.not_use_saved
    embedding_dict = get_embedding(use_saved=use_saved)