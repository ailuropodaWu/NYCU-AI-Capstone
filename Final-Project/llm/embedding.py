import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

from datasets import load_dataset
from torch.utils.data import DataLoader
from torchmetrics.functional.regression import spearman_corrcoef
from llama_cpp import Llama


torch.set_float32_matmul_precision("medium")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_ctx = 512

max_epochs = 3
batch_size = 4
learning_rate = 1e-5


class LLMEmbedding(pl.LightningModule):
    def __init__(self, llm: Llama, embedding_size=4096):
        super(LLMEmbedding, self).__init__()
        self.llm = llm
        self.seq = nn.Sequential(
            nn.Linear(embedding_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.scores = []
        self.similarities = []
        # self.save_hyperparameters()

    def forward(self, input: str):
        embeddings = torch.zeros((batch_size, n_ctx, 4096), device=device, dtype=torch.float32)
        for i, sentence in enumerate(input):
            e = torch.tensor(self.llm.embed(sentence), device=device, dtype=torch.float32, requires_grad=True)
            embeddings[i] += F.pad(e, (0, 0, 0, n_ctx - e.shape[0]))
        embeddings = self.seq(embeddings).mean(dim=1).unsqueeze(1)
        return embeddings

    def training_step(self, batch, batch_idx):
        embeddings_1 = self.forward(batch["sentence1"])
        embeddings_2 = self.forward(batch["sentence2"])
        for e1, e2 in zip(embeddings_1, embeddings_2):
            self.similarities.append(F.cosine_similarity(e1, e2).detach().cpu())
        similarities = torch.stack(self.similarities).to(device)
        self.scores.extend(batch["score"].detach().cpu() / 5)
        scores = torch.tensor(self.scores, device=device, dtype=torch.float32, requires_grad=True)
        correlations = spearman_corrcoef(similarities.squeeze(1), scores).type(torch.float32)
        loss = F.mse_loss(correlations, torch.tensor(1, device=device, dtype=torch.float32, requires_grad=True))
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        embeddings_1 = self.forward(batch["sentence1"])
        embeddings_2 = self.forward(batch["sentence2"])
        similarities = []
        for e1, e2 in zip(embeddings_1, embeddings_2):
            self.similarities.append(F.cosine_similarity(e1, e2).detach().cpu())
        similarities = torch.stack(self.similarities).to(device)
        self.scores.extend(batch["score"].detach().cpu() / 5)
        scores = torch.tensor(self.scores, device=device, dtype=torch.float32, requires_grad=True)
        correlations = spearman_corrcoef(similarities.squeeze(1), scores).type(torch.float32)
        loss = F.mse_loss(correlations, torch.tensor(1, device=device, dtype=torch.float32, requires_grad=True))
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=learning_rate)


def load_llm():
    return Llama(model_path="C:\\Users\\user\\.cache\\lm-studio\\models\\lmstudio-community\\Meta-Llama-3-8B-Instruct-BPE-fix-GGUF\\Meta-Llama-3-8B-Instruct-Q6_K.gguf", n_gpu_layers=-1, n_ctx=n_ctx, embedding=True, verbose=False)


if __name__ == "__main__":
    train_dataset = load_dataset("mteb/sts12-sts", split="train[:400]")
    test_dataset = load_dataset("mteb/sts12-sts", split="test[:40]")
    llm = load_llm()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, drop_last=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="llm-embedding-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min"
    )
    model = LLMEmbedding(llm)
    trainer = pl.Trainer(max_epochs=max_epochs, default_root_dir=os.getcwd(), callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, test_loader)
