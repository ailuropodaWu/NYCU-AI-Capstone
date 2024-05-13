from argparse import ArgumentParser
from similarity import *
from visualize_result import *
import os
import json

def main():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, help="Choose the directory of the model")
    parser.add_argument('--score_type', nargs='?', type=str, default='linear')
    parser.add_argument('--analyze', action="store_true", default=False)
    args = parser.parse_args()
    model = args.model
    score_type = args.score_type
    analyze = args.analyze
    embed_path = os.path.join(model, 'embedding_dict.json')
    pred_path = os.path.join(model, 'prediction.json')
    with open(embed_path, "r") as fp:
        embedding_dict = json.load(fp)
    semantic_similarity(embedding_dict, model, score_type, analyze)
    evaluation(pred_path)
    
    
if __name__ == '__main__':
    main()