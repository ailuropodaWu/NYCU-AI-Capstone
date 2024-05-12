import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import ArgumentParser

def visualize(prediction_path):
    def rmse(truth, pred):
        return np.sqrt(np.sum(np.square(truth - pred)) / truth.size)
    df = pd.read_json(prediction_path)
    truth = df['score']
    pred = df['similarity']
    
    print(f"RMSE: {rmse(truth, pred)}")
    
    for ran in range(5):
        t = truth[(ran <= truth) & (truth <= ran + 1)]
        p = pred[(ran <= truth) & (truth <= ran + 1)]
        print(f"[{ran} - {ran + 1}] RMSE: {rmse(t, p): .3f}, NUM: {t.size}")
    
    handle = [truth, pred]
    handle = np.array(handle)
    handle = handle.transpose()
    handle.sort(-1)
    handle = handle.transpose()
    plt.scatter(handle[0], handle[1])
    plt.show()
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Choose the directory of the model")
    args = parser.parse_args()
    model = args.model
    visualize(f'{model}/prediction.json')
    