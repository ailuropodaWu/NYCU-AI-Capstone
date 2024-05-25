import os
import json
import pandas as pd

from llama_cpp import Llama


data_type = "paragraph"

n_ctx = 512
n_ebd = 256


def load_llm():
    return Llama(model_path="C:\\Users\\user\\.cache\\lm-studio\\models\\lmstudio-community\\Meta-Llama-3-8B-Instruct-BPE-fix-GGUF\\Meta-Llama-3-8B-Instruct-Q6_K.gguf", n_gpu_layers=-1, n_ctx=n_ctx, embedding=True, verbose=False)


if __name__ == "__main__":
    llm = load_llm()
    dataset_path = f"../dataset/{data_type}_data.json"
    df = pd.read_json(dataset_path)[:100]
    data = set()
    llama_embedding = {}

    for _, row in df.iterrows():
        data.add(row[f"{data_type}1"])
        data.add(row[f"{data_type}2"])

    print(f"Total {len(data)} {data_type} to get embedding")

    for i, sentence in enumerate(data):
        llama_embedding[sentence] = llm.embed(sentence)
        if i % 100 == 0: print(f"Getting {i}th {data_type} embedding")

    with open(os.path.join(".", f"{data_type}_llama_embedding.json"), "w") as fp:
        json.dump(llama_embedding, fp)
