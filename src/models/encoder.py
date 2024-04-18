import torch
from torch import nn

from .attention import SelfAttention
from .feedforward import Feedforward


class EncoderAttentionLayer(nn.Module):
    def __init__(
        self,
        h: int,
        d_k: int,
        d_v: int,
        d_model: int,
        d_ff: int,
        dropout_p: float = 0.1,
        mask: bool = False,
    ):
        super().__init__()
        self.d_model = d_model

        self.attention = nn.Sequential(
            nn.LayerNorm(d_model),
            SelfAttention(h, d_k, d_v, d_model, mask=mask),
            nn.Dropout(dropout_p),
        )

        self.feedforward = nn.Sequential(
            nn.LayerNorm(d_model),
            Feedforward(d_model, d_ff),
            nn.Dropout(dropout_p),
        )

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_length, d_model)
        assert len(x.shape) == 3
        assert x.shape[2] == self.d_model

        x = x + self.attention(x)
        x = x + self.feedforward(x)

        assert x.shape[2] == self.d_model
        return x


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        h: int,
        d_k: int,
        d_v: int,
        d_model: int,
        d_ff: int,
        num_attention_layers: int,
        dropout_p: float = 0.1,
        mask: bool = False,
    ):
        super().__init__()
        self.d_model = d_model

        self.attention_layers = nn.ModuleList(
            [
                EncoderAttentionLayer(h, d_k, d_v, d_model, d_ff, dropout_p, mask=mask)
                for _ in range(num_attention_layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_length, d_model)
        assert len(x.shape) == 3
        batch_size, seq_length, _ = x.shape
        assert x.shape == (batch_size, seq_length, self.d_model)

        for attention_layer in self.attention_layers:
            x = attention_layer(x)
            assert x.shape == (batch_size, seq_length, self.d_model)

        return x
