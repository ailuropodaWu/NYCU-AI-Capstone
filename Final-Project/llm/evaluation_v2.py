import random
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformer_model import SentenceTransformerModel


random.seed(42)

model_name = "all-MiniLM-L12-v2"


def check_state_dict(m1, m2):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def evaluate():
    # model = SentenceTransformer(model_name, device="cuda")
    model = SentenceTransformerModel.load_from_checkpoint("lightning_logs/version_3/checkpoints/epoch=0-step=312.ckpt")
    model.eval()

    dataset = load_dataset("csv", data_files=["df_file.csv"], split="train")
    records = {label: [] for label in dataset.unique("Label")}
    results = []

    for data in tqdm(dataset, desc="Making evaluation data"):
        text, label = data["Text"], data["Label"]
        records[label].append(text)

    print(f"records distribution ({', '.join(f'{label}: {len(texts)}' for label, texts in records.items())})")

    for data in tqdm(dataset, desc="Evaluating model"):
        text, label = data["Text"], data["Label"]
        samples = [random.sample(texts, 1)[0] for texts in records.values()] + [text]
        embeddings = model.encode(samples, convert_to_tensor=True, device="cuda")
        similarities = [torch.cosine_similarity(embed, embeddings[[-1]]).cpu() for embed in embeddings[:-1]]
        pred = np.argmax(similarities).item()
        results.append((label, pred))

    df = pd.DataFrame(results, columns=["truth", "pred"])
    df.to_csv("evaluation.csv", index=False)
    print(f"accuracy: {(df['truth'] == df['pred']).sum() / len(df):.2%}")


if __name__ == "__main__":
    evaluate()
