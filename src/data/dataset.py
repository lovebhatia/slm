# src/data/dataset.py
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    PyTorch Dataset — loads tokenized text and returns
    (input, target) pairs for language model training.

    For every position i:
        input  = tokens[i : i + context_len]
        target = tokens[i+1 : i+1 + context_len]

    The model learns: given these tokens, predict the next one.
    """

    def __init__(self, token_ids: list, context_len: int):
        self.tokens      = torch.tensor(token_ids, dtype=torch.long)
        self.context_len = context_len

    def __len__(self):
        # How many samples can we make from this data?
        return len(self.tokens) - self.context_len

    def __getitem__(self, idx):
        # Input: tokens at positions idx → idx+context_len
        x = self.tokens[idx     : idx + self.context_len]
        # Target: tokens shifted one position forward
        y = self.tokens[idx + 1 : idx + self.context_len + 1]
        return x, y