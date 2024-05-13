import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from argparse import ArgumentParser

def evaluation(prediction_path):
    def rmse(score, pred):
        return np.sqrt(np.sum(np.square(score - pred)) / score.size)
    df = pd.read_json(prediction_path)
    score = df['score']
    cos_sim = df['cos-sim']
    tfidf_cos = df['tfidf-cos']
    print(pearsonr(score, cos_sim), spearmanr(score, cos_sim))
    print(pearsonr(tfidf_cos, cos_sim), spearmanr(tfidf_cos, cos_sim))
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Choose the directory of the model")
    args = parser.parse_args()
    model = args.model
    evaluation(f'{model}/prediction.json')