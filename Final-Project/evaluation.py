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
    print(
        f"""                   
                                 Pearson Spearman
            Semantic Similarity: {pearsonr(score, cos_sim)[0] * 100: .2f}  {spearmanr(score, cos_sim)[0] * 100: .2f}
            Lexical Similarity:  {pearsonr(tfidf_cos, cos_sim)[0] * 100: .2f}  {spearmanr(tfidf_cos, cos_sim)[0] * 100: .2f}
        """)
    
def main():
    parser = ArgumentParser()
    parser.add_argument('--model', nargs='*', type=str, help="Choose the directory of the model")
    parser.add_argument('--score_type', nargs='?', type=str, default='linear')
    parser.add_argument('--data_type', nargs='?', type=str, default='paragraph')
    parser.add_argument('--analyze', action="store_true", default=False)
    parser.add_argument('--no_calculate', action="store_true", default=False)
    
    model_list = ['e5', 'voyage', 'text_embedding', 'llm']
    args = parser.parse_args()
    models = args.model
    score_type = args.score_type
    data_type = args.data_type
    analyze = args.analyze
    no_calculate = args.no_calculate
    print(models)
    
    for model in models:
        assert model in model_list, f"{model} is invalid model"
        if not no_calculate:
            embed_path = os.path.join(model, f'{data_type}_embedding_dict.json')
            with open(embed_path, "r") as fp:
                embedding_dict = json.load(fp)
            semantic_similarity(embedding_dict, model, score_type, data_type, analyze)
        
        print(f"\t{model}:", end='')
        pred_path = os.path.join(model, f'{data_type}_prediction.json')
        evaluation(pred_path)
    
    
if __name__ == '__main__':
    main()