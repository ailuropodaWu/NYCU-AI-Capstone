import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L

from datasets import load_dataset
from torch.utils.data import DataLoader
from llama_cpp import Llama


torch.set_float32_matmul_precision("medium")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_ctx = 512
n_ebd = 256

max_epochs = 3
batch_size = 4
learning_rate = 1e-5


class LLMEmbedding(L.LightningModule):
    def __init__(self, llm: Llama, embedding_size=4096, temperature=1):
        super(LLMEmbedding, self).__init__()
        self.temperature = temperature
        self.llm = llm
        self.seq = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(embedding_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, n_ebd),
            nn.Tanh()
        )
        # self.save_hyperparameters()

    def train_dataloader(self):
        train_dataset = load_dataset("mteb/sts12-sts", split="train[:400]")
        return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        test_dataset = load_dataset("mteb/sts12-sts", split="test[:40]")
        return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, drop_last=True)

    def forward(self, input: str):
        embeddings = []
        for sentence in input:
            e = torch.tensor(self.llm.embed(sentence), device=device, requires_grad=True)
            embeddings.append(self.seq(F.pad(e, (0, 0, 0, n_ctx - e.shape[0]))))
        embeddings = torch.stack(embeddings).mean(dim=1)
        return embeddings

    def training_step(self, batch):
        embeddings = self.forward(batch["sentence1"])
        embeddings_plus = self.forward(batch["sentence1"])
        loss = torch.tensor(0, device=device, dtype=torch.float32, requires_grad=True)
        for i in range(batch_size):
            denominator = torch.tensor(0, device=device, dtype=torch.float32, requires_grad=True)
            for j in range(batch_size):
                denominator = denominator + torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[j]]) / self.temperature)[0]
            loss = loss - torch.log(torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[i]]) / self.temperature) / denominator)[0]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        embeddings = self.forward(batch["sentence1"])
        embeddings_plus = self.forward(batch["sentence1"])
        loss = torch.tensor(0, device=device, dtype=torch.float32, requires_grad=True)
        for i in range(batch_size):
            denominator = torch.tensor(0, device=device, dtype=torch.float32, requires_grad=True)
            for j in range(batch_size):
                denominator = denominator + torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[j]]) / self.temperature)[0]
            loss = loss - torch.log(torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[i]]) / self.temperature) / denominator)[0]
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=learning_rate)


def load_llm():
    return Llama(model_path="C:\\Users\\user\\.cache\\lm-studio\\models\\lmstudio-community\\Meta-Llama-3-8B-Instruct-BPE-fix-GGUF\\Meta-Llama-3-8B-Instruct-Q6_K.gguf", n_gpu_layers=-1, n_ctx=n_ctx, embedding=True, verbose=False)


if __name__ == "__main__":
    llm = load_llm()
    model = LLMEmbedding(llm)
    trainer = L.Trainer(max_epochs=max_epochs, default_root_dir=os.getcwd())
    trainer.fit(model)
