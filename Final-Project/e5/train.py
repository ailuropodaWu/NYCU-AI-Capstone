import torch
import torch.nn.functional as F
import lightning as pl
from torch import Tensor
from sentence_transformers import SentenceTransformer
from prepare_data import DataModule


class e5Embedding(pl.LightningModule):
    def __init__(self, temperature=0.5, learning_rate=1e-4):
        super().__init__()
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.encoder = SentenceTransformer('intfloat/e5-large-v2')
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(1024, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )

    def forward(self, x):
        outputs = self.encoder.encode([f"query: {sentence}" for sentence in x], normalize_embeddings=True, convert_to_tensor=True, device=self.device)
        return outputs
    
    def info_nce_loss(self, batch):
        embeddings = self.forward(batch)
        embeddings_plus = self.forward(batch)
        embeddings = self.projection(embeddings)
        embeddings_plus = self.projection(embeddings_plus)
        
        loss = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
        for i in range(len(embeddings)):
            denominator = torch.tensor(0, device=self.device, dtype=torch.float32, requires_grad=True)
            for j in range(len(embeddings)):
                denominator = denominator + torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[j]]) / self.temperature).squeeze()
            loss = loss - torch.log(torch.exp(F.cosine_similarity(embeddings[[i]], embeddings_plus[[i]]) / self.temperature) / denominator).squeeze()
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.info_nce_loss(batch)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.info_nce_loss(batch)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    model = e5Embedding(temperature=0.05, learning_rate=1e-5)
    trainer = pl.Trainer(max_epochs=4)
    data_module = DataModule(batch_size=6)
    trainer.fit(model, data_module)