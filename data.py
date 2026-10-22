"""data.py — packed token dataset + byte-level fallback tokenizer.

Developers   : PAI AND Yoon A Limsuwan
Organization : MSPS NETWORK
Licence      : MIT
Year         : 2026
"""

from __future__ import annotations

import os
import torch
from torch.utils.data import Dataset


class PackedTokenDataset(Dataset):
    """Pre-tokenized corpus packed into fixed-length (seq_len) samples.

    Production: replace with mosaicml-streaming / HF datasets.
    """

    def __init__(self, path: str, seq_len: int = 256, cache: str = None):
        self.seq_len = seq_len
        if cache and os.path.exists(cache):
            self.tokens = torch.load(cache)
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            self.tokens = torch.frombuffer(
                bytearray(text.encode("utf-8")), dtype=torch.uint8
            ).long()
            if cache:
                torch.save(self.tokens, cache)
        n = len(self.tokens) // (seq_len + 1)
        self.n = max(n, 1)
        print(f"[data] {len(self.tokens):,} bytes -> {self.n:,} samples")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        chunk = self.tokens[i * (self.seq_len + 1):(i + 1) * (self.seq_len + 1)]
        if len(chunk) < self.seq_len + 1:
            chunk = torch.cat([
                chunk,
                torch.zeros(self.seq_len + 1 - len(chunk), dtype=torch.long),
            ])
        return chunk[:-1].clone(), chunk[1:].clone()


class ByteDetokenizer:
    def decode(self, ids):
        return bytes([max(0, min(255, int(i)))
                      for i in ids]).decode("utf-8", errors="ignore")
