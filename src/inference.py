import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from typing import Literal

from .bpe import SimpleBytePairEncoding


class TextGenerator:
    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        initial_tokens: list[int] | None = None,
        device: Literal["cpu", "mps", "cuda"] = "cpu",
    ):
        model.eval()

        self.model = model
        self.vocab_size = vocab_size
        self.device = device

        self.sentinel = vocab_size

        if initial_tokens is None:
            initial_tokens = []
        self.tokens = initial_tokens

    def __iter__(self):
        return self

    def __next__(self):
        x = torch.tensor(self.tokens, device=self.device).unsqueeze(0)
        assert x.shape == (1, len(self.tokens))

        log_preds = F.log_softmax(self.model(x), dim=2)
        assert log_preds.shape == (1, len(self.tokens), self.vocab_size + 1)

        # only want last token's prediction for inference
        log_preds = log_preds[0, -1, :]
        assert log_preds.shape == (self.vocab_size + 1,)

        log_preds = log_preds.to("cpu").detach().numpy()
        preds = np.exp(log_preds)
        token_pred = np.random.choice(len(preds), p=preds)
        self.tokens.append(token_pred)
        return token_pred


def generate_text(
    prompt: str,
    num_tokens: int,
    model: nn.Module,
    vocab_size: int,
    bpe: SimpleBytePairEncoding,
    device: Literal["cpu", "mps", "cuda"] = "cpu",
) -> str:
    if len(prompt) == 0:
        raise ValueError("Prompt must be non-empty!")

    initial_tokens = bpe.encode(prompt, visualise=None)

    generator = TextGenerator(
        model=model,
        vocab_size=vocab_size,
        initial_tokens=initial_tokens,
        device=device,
    )

    print(prompt, end="")

    result = ""
    for _, token in zip(range(num_tokens), generator):
        # print(token)
        word = bpe.decode([token])
        print(word, end="")
        result += word

    return result
