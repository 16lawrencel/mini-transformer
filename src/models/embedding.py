import torch
from torch import nn

import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 1000,
    ):
        super().__init__()
        assert d_model % 2 == 0
        self.d_model = d_model
        self.encoding_constant = 10000

        position_encodings = self._compute_position_encodings(d_model, max_len)
        # register buffer so position_encodings is moved to appropriate device
        # when model is moved to device
        self.register_buffer("position_encodings", position_encodings)

    def _compute_position_encodings(self, d_model: int, seq_length: int):
        encodings_per_dimension = []
        for i in range(d_model // 2):
            seq_tensor = torch.arange(seq_length)
            encodings_in_sinusoid = seq_tensor / (
                self.encoding_constant ** (2 * i / d_model)
            )
            encodings_per_dimension.append(torch.sin(encodings_in_sinusoid))
            encodings_per_dimension.append(torch.cos(encodings_in_sinusoid))

        assert len(encodings_per_dimension) == d_model

        position_encodings = torch.stack(encodings_per_dimension).permute(1, 0)
        assert position_encodings.shape == (seq_length, d_model)

        return position_encodings

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_length, d_model)
        assert len(x.shape) == 3
        assert x.shape[2] == self.d_model

        seq_length = x.shape[1]

        return self.position_encodings[:seq_length].unsqueeze(0)


class PositionAwareEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncoding(d_model)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_length)
        assert len(x.shape) == 2

        x = self.embedding(x) * np.sqrt(self.d_model)
        x = x + self.position_encoding(x)

        return x
