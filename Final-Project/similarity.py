import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity



def cosine2score(cos_sim, score_type='linear'):
    assert score_type in ['linear', 'exp']
    linear_baseline = 0.85
    if score_type == 'linear':
        score = (cos_sim - linear_baseline) / (1 - linear_baseline)
    elif score_type == 'exp':
        score = np.exp(cos_sim - 1)
    return round(min(max(score, 0), 1) * 5, 1)

def semantic_similarity(embedding_dict, model, score_type='linear', data_type='sentence', analyze=False):
    
    dataset_path = f"./dataset/{data_type}_data.json"
    df = pd.read_json(dataset_path)
    pred_path = os.path.join(model, f"{data_type}_prediction.json")
    similarity_list = []
    predictions = []
    print(f"Embedding dim: {len(embedding_dict[df[f"{data_type}1"][0]])}")
    for _, row in tqdm(df.iterrows()):
        sen1, sen2 = row[f"{data_type}1"], row[f"{data_type}2"]
        similarity = cosine_similarity([embedding_dict[sen1]], [embedding_dict[sen2]]).flatten()
        similarity_list.append(similarity)
        pred = list(row.values)
        pred.append(similarity[0])
        predictions.append(pred)
    if analyze:
        similarity_list = np.array(similarity_list)
        print(f"Max: {similarity_list.max()}, Min: {similarity_list.min()}, Mean: {similarity_list.mean()}")
        similarity_list = np.round(similarity_list, 2)
        count = np.unique(similarity_list, return_index=True, return_counts=True)
        x, h = count[0], count[2]
        plt.bar(x, h)
        plt.show()
        
    predictions = pd.DataFrame(predictions, columns=[f"{data_type}1", f"{data_type}2", "score", "tfidf-cos", "cos-sim"])
    print(predictions)
    predictions.to_json(pred_path, orient="records")
    return predictions
