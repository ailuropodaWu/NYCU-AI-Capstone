import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from argparse import ArgumentParser

def visualize(prediction_path):
    df = pd.read_json(prediction_path)
    handle = [df['score'], df['similarity']]
    handle = np.array(handle)
    handle = handle.transpose()
    handle.sort(-1)
    handle = handle.transpose()
    print(handle)
    plt.scatter(handle[0], handle[1])
    plt.show()
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, help="Choose the directory of the model")
    args = parser.parse_args()
    model = args.model
    visualize(f'{model}/prediction.json')
    