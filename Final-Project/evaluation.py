from argparse import ArgumentParser
from similarity import *
import os
import json
from scipy.stats import pearsonr, spearmanr

def evaluation(prediction_path):
    def rmse(score, pred):
        return np.sqrt(np.sum(np.square(score - pred)) / score.size)
    df = pd.read_json(prediction_path)
    score = df['score']
    cos_sim = df['cos-sim']
    tfidf_cos = df['tfidf-cos']
    print(pearsonr(score, cos_sim), spearmanr(score, cos_sim))
    print(pearsonr(tfidf_cos, cos_sim), spearmanr(tfidf_cos, cos_sim))
    
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