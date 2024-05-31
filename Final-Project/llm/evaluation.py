import numpy as np
from torch.nn.functional import cosine_similarity, cross_entropy
from torch import Tensor
from tqdm import tqdm
from ..dataset.load_eval_data import load_eval_data


def evaluation(model):
    truth = []
    pred = []
    dataset = load_eval_data()
    for _, data in tqdm(dataset.iterrows(), total=len(dataset)):
        embeddings = model(data.values[:-1])
        similarity = [cosine_similarity(embed, embeddings[[-1]]).cpu() for embed in embeddings[:-1]]
        truth.append(data['label'])
        pred.append(np.argmax(similarity))
    return cross_entropy(Tensor(pred), Tensor(truth)).item(), Tensor(pred) == Tensor(truth) / len(truth)
