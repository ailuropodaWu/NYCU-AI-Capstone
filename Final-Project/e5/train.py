import torch
import torch.nn.functional as F
import lightning as pl
from torch import Tensor
from transformers import AutoModel
from prepare_data import DataModule

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class e5Embedding(pl.LightningModule):
    def __init__(self, temperature=0.5, learning_rate=1e-4):
        super().__init__()
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.encoder = AutoModel.from_pretrained('intfloat/e5-large-v2')
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.config.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128)
        )

    def forward(self, x):
        outputs = self.encoder(**x)
        embeddings = average_pool(outputs.last_hidden_state, x['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
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
    model = e5Embedding()
    trainer = pl.Trainer(max_epochs=2)
    data_module = DataModule(batch_size=4)
    trainer.fit(model, data_module)