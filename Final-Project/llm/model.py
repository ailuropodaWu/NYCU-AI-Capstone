import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L

from torch.utils.data import DataLoader
from llama_cpp import Llama

from dataset import LLMEmbeddingDataset


class LLMEmbeddingModel(L.LightningModule):
    def __init__(self, batch_size=4, llm_emb=4096, n_ctx=512, n_emb=256, learning_rate=1e-5, temperature=0.05, dataset_path="../dataset/extended_data.json"):
        super(LLMEmbeddingModel, self).__init__()
        self.batch_size = batch_size
        self.n_ctx = n_ctx
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.dataset_path = dataset_path
        self.seq = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(llm_emb, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_emb),
            nn.Tanh()
        )

        self.save_hyperparameters()

        self.llm = Llama(model_path="C:\\Users\\user\\.cache\\lm-studio\\models\\lmstudio-community\\Meta-Llama-3-8B-Instruct-BPE-fix-GGUF\\Meta-Llama-3-8B-Instruct-Q6_K.gguf", n_gpu_layers=-1, n_ctx=n_ctx, embedding=True, verbose=False)

    def train_dataloader(self):
        train_dataset = LLMEmbeddingDataset(dataset_path=self.dataset_path, mode="train")
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        test_dataset = LLMEmbeddingDataset(dataset_path=self.dataset_path, mode="test")
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, drop_last=True)

    def forward(self, inputs):
        embeddings = []
        for sentence in inputs:
            e = torch.tensor(self.llm.embed(sentence), device=self.device, requires_grad=True)
            embeddings.append(self.seq(F.pad(e, (0, 0, 0, self.n_ctx - e.shape[0]))))
        embeddings = torch.stack(embeddings).mean(dim=1)
        return embeddings

    def training_step(self, batch):
        embeddings = self.forward(batch)
        embeddings_plus = self.forward(batch)
        loss = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
        for i in range(self.batch_size):
            denominator = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
            for j in range(self.batch_size):
                denominator = denominator + torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[j]]) / self.temperature).squeeze()
            loss = loss - torch.log(torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[i]]) / self.temperature) / denominator).squeeze()
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch):
        embeddings = self.forward(batch)
        embeddings_plus = self.forward(batch)
        loss = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
        for i in range(self.batch_size):
            denominator = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
            for j in range(self.batch_size):
                denominator = denominator + torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[j]]) / self.temperature)[0]
            loss = loss - torch.log(torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[i]]) / self.temperature) / denominator)[0]
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.learning_rate)
