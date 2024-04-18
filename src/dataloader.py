import torch
from torch.utils.data import Dataset, DataLoader
from tiktoken._educational import SimpleBytePairEncoding

from pathlib import Path
from IPython.utils import io


class TinyShakespeareDataset(Dataset):
    def __init__(
        self,
        path: Path,
        bpe: SimpleBytePairEncoding,
        split_length: int,
        preprocess: bool = False,
    ):
        super(Dataset).__init__()
        self.path = path

        with path.open() as f:
            self.text = f.read()

        if preprocess:
            self.text = self.text.replace("\n", " ")
            self.text = self.text.lower()

        with io.capture_output():
            self.tokens = bpe.encode(self.text)

        self.split_length = split_length
        self.length = len(self.tokens) // split_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        start = idx * self.split_length
        end = (idx + 1) * self.split_length
        return torch.tensor(self.tokens[start:end], dtype=torch.int32)


def get_dataloader(
    data_path: Path,
    bpe: SimpleBytePairEncoding,
    split_length: int = 100,
    batch_size: int = 10,
    shuffle: bool = True,
    preprocess: bool = False,
) -> DataLoader:
    dataset = TinyShakespeareDataset(
        data_path, bpe, split_length=split_length, preprocess=preprocess
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
