import os

import torch
import lightning as L

from model import LLMEmbeddingModel


torch.set_float32_matmul_precision("medium")


max_epochs = 1


if __name__ == "__main__":
    model = LLMEmbeddingModel(batch_size=4, llm_emb=4096, n_ctx=512, n_emb=1024, learning_rate=1e-5, temperature=0.05, dataset_path="../dataset/paragraph_data.json")
    trainer = L.Trainer(max_epochs=max_epochs, default_root_dir=os.getcwd())
    trainer.fit(model)
