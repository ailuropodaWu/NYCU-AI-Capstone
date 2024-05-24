import json

from torch.utils.data import Dataset


class LLMEmbeddingDataset(Dataset):
    def __init__(self, dataset_path: str, mode: str = "train"):
        self.dataset_path = dataset_path
        self.mode = mode
        self.paragraphs = []

        self.init_dataset()

    def init_dataset(self):
        with open(self.dataset_path, "r") as fp:
            self.paragraphs = [data["paragraph1"] for data in json.load(fp)][:4000]

        if self.mode == "train": self.paragraphs = self.paragraphs[:int(0.8 * len(self.paragraphs))]
        elif self.mode == "test": self.paragraphs = self.paragraphs[int(0.8 * len(self.paragraphs)):]
        else: raise ValueError("Invalid mode")

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        return self.paragraphs[idx]
