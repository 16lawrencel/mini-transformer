import torch
from torch import nn

from .embedding import PositionAwareEmbedding
from .encoder import EncoderTransformer

total_times = []


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        h: int,
        d_model: int,
        d_ff: int,
        num_attention_layers: int,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        assert d_model % h == 0
        d_k = d_model // h
        d_v = d_model // h

        # We want to share embedding layer between encoder and decoder
        self.position_aware_embedding = PositionAwareEmbedding(vocab_size, d_model)

        self.encoder_transformer = EncoderTransformer(
            h,
            d_k,
            d_v,
            d_model,
            d_ff,
            num_attention_layers,
            dropout_p,
            mask=True,
        )

        self.feature_map = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        # x: (batch_size, seq_length) - input
        assert len(x.shape) == 2
        batch_size, seq_length = x.shape

        x = self.position_aware_embedding(x)
        x = self.encoder_transformer(x)
        assert x.shape == (batch_size, seq_length, self.d_model)

        out = self.feature_map(x)
        assert out.shape == (batch_size, seq_length, self.vocab_size)

        return out
