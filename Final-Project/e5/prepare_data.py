import json
import lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class CustomDataset(Dataset):
    def __init__(self, texts, transform=None):
        self.texts = texts
        self.transform = transform
        self.tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
        for key in inputs.keys():
            inputs[key] = inputs[key].squeeze(0)
        if self.transform:
            inputs = self.transform(inputs)
        return inputs

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size=32):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.init_dataset()

    def init_dataset(self):
        with open(self.dataset_path, "r") as fp:
            self.paragraphs = [data["paragraph1"] for data in json.load(fp)][:100]

        self.train_texts = self.paragraphs[:int(0.8 * len(self.paragraphs))]
        self.val_texts = self.paragraphs[int(0.8 * len(self.paragraphs)):]

    def setup(self, stage=None):
        self.train_dataset = CustomDataset(self.train_texts)
        self.val_dataset = CustomDataset(self.val_texts)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

