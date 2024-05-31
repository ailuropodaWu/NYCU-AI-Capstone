import json

from torch.utils.data import Dataset


class LLMEmbeddingDataset(Dataset):
    def __init__(self, dataset_path: str, mode: str = "train"):
        self.dataset_path = dataset_path
        self.mode = mode
        self.paragraphs = []

        self.init_dataset()

    def init_dataset(self):
        excluded = 5000
        with open(self.dataset_path, "r") as fp:
            dataset = json.load(fp)
            self.paragraphs.extend([data["paragraph1"] for data in dataset][:-excluded])
        self.paragraphs = list(set(self.paragraphs))

        if self.mode == "train": self.paragraphs = self.paragraphs[:-1000]
        elif self.mode == "test": self.paragraphs = self.paragraphs[-1000:]
        else: raise ValueError("Invalid mode")

    def __len__(self):
        return len(self.paragraphs)

    def __getitem__(self, idx):
        return self.paragraphs[idx]
