import torch
import torch.nn.functional as F
import lightning as pl
from argparse import ArgumentParser
from torch import nn
from prepare_data import load_data, CustomDataModule

class SimCLR(pl.LightningModule):
    def __init__(self, input_dim, projection_dim=128, temperature=0.5, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.temperature = temperature
        self.learning_rate = learning_rate
        assert input_dim in [1536], f"Input dim: {input_dim}. Check embedding dimension from embedding models "

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
            nn.ReLU()
        )
        self.projection = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return F.normalize(self.encoder(x), dim=1)

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

# Training script
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch', nargs='?', type=int, default=4)
    parser.add_argument('--epoch', nargs='?', type=int, default=2)
    args = parser.parse_args()
    batch_size = args.batch
    epoch = args.epoch
    df = load_data()
    input_dim = df['embedding'].iloc[0].shape[0]
    data_module = CustomDataModule(df, batch_size)
    model = SimCLR(input_dim=input_dim, temperature=0.05, learning_rate=1e-5)
    trainer = pl.Trainer(max_epochs=epoch)
    
    print(f"Epochs: {epoch} / Batch Size: {batch_size} / Input Dim: {input_dim}")
    trainer.fit(model, datamodule=data_module)
