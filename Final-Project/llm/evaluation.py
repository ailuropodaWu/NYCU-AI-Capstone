import numpy as np
from torch.nn.functional import cosine_similarity, cross_entropy
from torch import Tensor
from ..dataset.load_eval_data import load_eval_data


def evaluation(model):
    truth = []
    pred = []
    dataset = load_eval_data()
    for _, data in dataset.iterrows():
        embedding = [model(sentence) for sentence in data.values[:-1]]
        similarity = [cosine_similarity(embed, embedding[-1]) for embed in embedding[:-1]]
        truth.append(data['label'])
        pred.append(np.argmax(similarity))
        truth = Tensor(truth)
        pred = Tensor(pred)
    return cross_entropy(pred, truth)

