import voyageai
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import json
from argparse import ArgumentParser
dataset_path = "../dataset/data.json"
df = pd.read_json(dataset_path)


def get_embedding(save_path="embedding_dict.json", use_saved=True):
    
    if use_saved:
        with open("embedding_dict.json", "r") as fp:
            embedding_dict = json.load(fp)
        return embedding_dict
    
    api_key = "pa-V6akNHc5VxAm-m5bSRWBjmfc260_9sArATPfJxeCkGM"
    vo = voyageai.Client(api_key=api_key)
    
    embedding_dict = {}
    sentences = set()
    batch = []
    batch_cnt = 0
    batch_size = 128
    for _, row in tqdm(df.iterrows()):
        sentences.add(row["sentence1"])
        sentences.add(row["sentence2"])

    for sentence in tqdm(sentences):
        embedding_dict[sentence] = batch_cnt
        batch.append(sentence)
        batch_cnt += 1
        if batch_cnt == batch_size:
            batch_cnt = 0
            embeddings = vo.embed(batch, "voyage-2", input_type="document").embeddings
            for s in batch:
                embedding_dict[s] = embeddings[embedding_dict[s]]
            batch = []
            time.sleep(0.1)
    if len(batch) > 0:
        embeddings = vo.embed(batch, "voyage-2", input_type="document").embeddings
        for s in batch:
            embedding_dict[s] = embeddings[embedding_dict[s]]
    with open("embedding_dict.json", "w") as fp:
        json.dump(embedding_dict, fp)
    return embedding_dict

def predict_similarity(embedding_dict):
    predictions = []
    for _, row in tqdm(df.iterrows()):
        sen1, sen2 = row["sentence1"], row["sentence2"]
        similarity = cosine_similarity([embedding_dict[sen1]], [embedding_dict[sen2]])
        pred = list(row.values)
        pred.append(similarity.flatten()[0] * 5)
        predictions.append(pred)
    predictions = pd.DataFrame(predictions, columns=["sentence1", "sentence2", "score", "similarity"])
    print(predictions)
    predictions.to_json("./prediction.json", orient="records")
    return predictions

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--not_use_saved", action="store_true", default=False)
    args = parser.parse_args()
    use_saved = not args.not_use_saved
    embedding_dict = get_embedding(use_saved=use_saved)
    preidictions = predict_similarity(embedding_dict)