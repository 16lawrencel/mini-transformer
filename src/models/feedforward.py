import torch
from torch import nn


class Feedforward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
    ):
        super().__init__()
        self.d_model = d_model

        self.feedforward_1 = nn.Linear(d_model, d_ff, bias=True)
        self.relu = nn.ReLU()
        self.feedforward_2 = nn.Linear(d_ff, d_model, bias=True)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_length, d_model)
        assert len(x.shape) == 3
        assert x.shape[2] == self.d_model

        x = self.relu(self.feedforward_1(x))
        x = self.feedforward_2(x)

        assert x.shape[2] == self.d_model

        return x
