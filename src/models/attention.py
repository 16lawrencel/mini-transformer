import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules import MultiheadAttention as TorchMultiheadAttention

import numpy as np


class Attention(nn.Module):
    def __init__(self, d_k: int, d_v: int, h: int, mask: bool = False):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.mask = mask

    def _autoregressive_mask(self, softmax_inp: torch.Tensor):
        batch_size, _, seq_length, _ = softmax_inp.shape
        assert softmax_inp.shape == (batch_size, self.h, seq_length, seq_length)

        mask = (
            torch.triu(
                torch.ones(seq_length, seq_length, device=softmax_inp.device),
                diagonal=1,
            )
            .view(1, 1, seq_length, seq_length)
            .bool()
        )

        ret = softmax_inp.masked_fill_(mask, -float("inf"))
        return ret

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        # Q: (batch_size, h, seq_length, d_k)
        # K: (batch_size, h, seq_length, d_k)
        # V: (batch_size, h, seq_length, d_v)
        assert len(Q.shape) == 4
        assert len(K.shape) == 4
        assert len(V.shape) == 4

        batch_size = Q.shape[0]
        seq_length = Q.shape[2]
        assert Q.shape == (batch_size, self.h, seq_length, self.d_k)
        assert K.shape == (batch_size, self.h, seq_length, self.d_k)
        assert V.shape == (batch_size, self.h, seq_length, self.d_v)

        K_T = torch.transpose(K, 3, 2)
        softmax_inp = Q @ K_T / np.sqrt(self.d_k)
        if self.mask:
            softmax_inp = self._autoregressive_mask(softmax_inp)

        result = F.softmax(softmax_inp, dim=3) @ V
        assert result.shape == (batch_size, self.h, seq_length, self.d_v)

        return result


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        h: int,
        d_k: int,
        d_v: int,
        d_model: int,
        mask: bool = False,
    ):
        super().__init__()
        assert d_k == d_v
        assert h * d_k == d_model

        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.mask = mask

        self.proj_Q = nn.Linear(d_model, d_k * h, bias=False)
        self.proj_K = nn.Linear(d_model, d_k * h, bias=False)
        self.proj_V = nn.Linear(d_model, d_v * h, bias=False)
        self.proj_O = nn.Linear(d_v * h, d_model)

        self.attention = Attention(d_k, d_v, h, mask)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        # Q: (batch_size, seq_length, d_k)
        # K: (batch_size, seq_length, d_k)
        # V: (batch_size, seq_length, d_v)
        assert len(Q.shape) == 3
        assert len(K.shape) == 3
        assert len(V.shape) == 3

        batch_size = Q.shape[0]
        seq_length = Q.shape[1]
        assert Q.shape == (batch_size, seq_length, self.d_model)
        assert K.shape == (batch_size, seq_length, self.d_model)
        assert V.shape == (batch_size, seq_length, self.d_model)

        # Do batch computation for all the attention heads
        proj_Q = (
            self.proj_Q(Q)
            .view(batch_size, seq_length, self.h, self.d_k)
            .transpose(1, 2)
        )
        proj_K = (
            self.proj_K(K)
            .view(batch_size, seq_length, self.h, self.d_k)
            .transpose(1, 2)
        )
        proj_V = (
            self.proj_V(V)
            .view(batch_size, seq_length, self.h, self.d_v)
            .transpose(1, 2)
        )

        assert proj_Q.shape == (batch_size, self.h, seq_length, self.d_k)
        assert proj_K.shape == (batch_size, self.h, seq_length, self.d_k)
        assert proj_V.shape == (batch_size, self.h, seq_length, self.d_v)

        head_results = self.attention(
            proj_Q,
            proj_K,
            proj_V,
        )
        assert head_results.shape == (batch_size, self.h, seq_length, self.d_v)

        head_results = head_results.transpose(1, 2).reshape(
            batch_size, seq_length, self.h * self.d_v
        )

        result = self.proj_O(head_results)

        return result


class SelfAttention(nn.Module):
    """
    Thin wrapper around MultiheadAttention.
    SelfAttention calls MultiheadAttention and passes x for
    the Q, K, and V vectors.
    """

    def __init__(
        self,
        h: int,
        d_k: int,
        d_v: int,
        d_model: int,
        mask: bool = False,
    ):
        super().__init__()
        assert d_k == d_v
        assert h * d_k == d_model

        self.multihead_attention = MultiheadAttention(
            h,
            d_k,
            d_v,
            d_model,
            mask=mask,
        )

    def forward(self, x: torch.Tensor):
        return self.multihead_attention(x, x, x)
