import pandas as pd
import time
from tqdm import tqdm
import json
from openai import OpenAI
from argparse import ArgumentParser


def get_embedding(save_path="embedding_dict.json", use_saved=True):
    dataset_path = "./dataset/data.json"
    df = pd.read_json(dataset_path)
    print("Getting Embedding ...")
    
    if use_saved:
        with open(save_path, "r") as fp:
            embedding_dict = json.load(fp)
        return embedding_dict
    api_key = None
    client = OpenAI(organization="org-HYkhp068aByFBQ1iIDHMMvDY", api_key=api_key)
    
    
    embedding_dict = {}
    sentences = set()
    batch = []
    batch_cnt = 0
    batch_size = 256
    for _, row in tqdm(df.iterrows()):
        sentences.add(row["sentence1"])
        sentences.add(row["sentence2"])

    for sentence in tqdm(sentences):
        embedding_dict[sentence] = batch_cnt
        batch.append(sentence)
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