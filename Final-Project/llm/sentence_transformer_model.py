import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lightning as L

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from datasets import load_dataset


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


class SentenceTransformerModel(L.LightningModule):
    def __init__(self, batch_size=4, n_ctx=512, n_emb=256, learning_rate=1e-5, temperature=0.05):
        super(SentenceTransformerModel, self).__init__()
        self.batch_size = batch_size
        self.n_ctx = n_ctx
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.lm = SentenceTransformer("intfloat/e5-large-v2", device=self.device)
        self.mlp = nn.Sequential(
            nn.Linear(n_emb, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.save_hyperparameters()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_save_checkpoint(self, checkpoint) -> None:
        torch.save(self.lm.state_dict(), "./e5-large-v2.pth")
        return super().on_save_checkpoint(checkpoint)

    def train_dataloader(self):
        train_dataset = load_dataset("wikipedia", "20220301.simple", split="train[:5000]")
        print(f"train dataset size: {len(train_dataset)}")
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)

    def val_dataloader(self):
        test_dataset = load_dataset("wikipedia", "20220301.simple", split="train[5000:5500]")
        print(f"test dataset size: {len(test_dataset)}")
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, persistent_workers=True, drop_last=True)

    def forward(self, inputs):
        embeddings = self.lm.encode([f"query: {sentence}" for sentence in inputs], normalize_embeddings=True, convert_to_tensor=True, device=self.device)
        # embeddings = self.seq(embeddings)
        return embeddings

    def training_step(self, batch):
        inputs = batch["text"]
        embeddings = self.mlp(self.forward(inputs))
        embeddings_plus = self.mlp(self.forward(inputs))
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
        embeddings = self.mlp(self.forward(inputs))
        embeddings_plus = self.mlp(self.forward(inputs))
        loss = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
        for i in range(self.batch_size):
            denominator = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
            for j in range(self.batch_size):
                denominator = denominator + torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[j]]) / self.temperature).squeeze()
            loss = loss - torch.log(torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[i]]) / self.temperature) / denominator).squeeze()
        self.log("val_loss", loss, prog_bar=True, batch_size=self.batch_size)
        return loss


if __name__ == "__main__":
    model = SentenceTransformerModel(batch_size=16, n_ctx=512, n_emb=1024, learning_rate=1e-5, temperature=0.03)
    trainer = L.Trainer(max_epochs=1, default_root_dir=os.getcwd(), log_every_n_steps=5)
    trainer.fit(model)

    # dataset = load_dataset("wikipedia", "20220301.simple", split="train[:5000]")
    # print(dataset[0], dataset[1], dataset[2])

    model = SentenceTransformerModel("intfloat/e5-large-v2").to("cuda")
    model.load_state_dict(torch.load("./e5-large-v2.pth"), strict=False)
    model.eval()

    dataset = load_dataset("csv", data_files=["df_file.csv"], split="train")

    for data in dataset:
        text, label = data["Text"], data["Label"]
        embeddings = model.forward(text)
        print(embeddings, label)
        break
