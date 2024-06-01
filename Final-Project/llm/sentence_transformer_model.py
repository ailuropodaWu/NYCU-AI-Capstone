import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class SentenceTransformerModel(L.LightningModule):
    def __init__(self, batch_size=4, n_ctx=512, n_emb=1024, learning_rate=1e-5, temperature=0.05):
        super(SentenceTransformerModel, self).__init__()
        self.model_name = "all-MiniLM-L12-v2"

        self.batch_size = batch_size
        self.n_ctx = n_ctx
        self.learning_rate = learning_rate
        self.temperature = temperature

        self.embedding_model = SentenceTransformer(self.model_name, device=self.device)
        self.embedding_model.max_seq_length = self.n_ctx
        self.embedding_model.requires_grad_(False)
        self.embedding_model.eval()

        self.seq = nn.Sequential(
            nn.Linear(384, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, n_emb)
        )

        self.save_hyperparameters()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def on_save_checkpoint(self, checkpoint) -> None:
        torch.save(self.embedding_model.state_dict(), f"{self.model_name}.pth")
        return super().on_save_checkpoint(checkpoint)

    def train_dataloader(self):
        train_dataset = load_dataset("wikipedia", "20220301.simple", split="train[:20000]")
        print(f"train dataset size: {len(train_dataset)}")
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        test_dataset = load_dataset("wikipedia", "20220301.simple", split="train[20000:20500]")
        print(f"test dataset size: {len(test_dataset)}")
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, drop_last=True)

    def forward(self, inputs):
        embeddings = self.embedding_model.encode(inputs, normalize_embeddings=True, convert_to_tensor=True, device=self.device)
        embeddings = self.seq(embeddings)
        return embeddings

    def encode(self, inputs, normalize_embeddings=True, convert_to_tensor=True, device=None):
        return self.embedding_model.encode(inputs, normalize_embeddings=normalize_embeddings, convert_to_tensor=convert_to_tensor, device=device)

    def training_step(self, batch):
        inputs = batch["text"]
        embeddings = self.forward(inputs)
        embeddings_plus = self.forward(inputs)
        loss = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
        for i in range(self.batch_size):
            denominator = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
            for j in range(self.batch_size):
                denominator = denominator + torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[j]]) / self.temperature).squeeze()
            loss = loss - torch.log(torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[i]]) / self.temperature) / denominator).squeeze()
        self.log("train_loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch):
        inputs = batch["text"]
        embeddings = self.forward(inputs)
        embeddings_plus = self.forward(inputs)
        loss = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
        for i in range(self.batch_size):
            denominator = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
            for j in range(self.batch_size):
                denominator = denominator + torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[j]]) / self.temperature).squeeze()
            loss = loss - torch.log(torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[i]]) / self.temperature) / denominator).squeeze()
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss


if __name__ == "__main__":
    model = SentenceTransformerModel(batch_size=64, n_ctx=512, n_emb=1024, learning_rate=1e-5, temperature=0.05)
    trainer = L.Trainer(max_epochs=1, default_root_dir=os.getcwd(), log_every_n_steps=5)
    trainer.fit(model)
