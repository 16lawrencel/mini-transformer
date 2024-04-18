import pickle
from pathlib import Path
from tiktoken._educational import SimpleBytePairEncoding

from IPython.utils import io


def _train_bpe(text: str, vocab_size: int) -> SimpleBytePairEncoding:
    gpt2_pattern = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with io.capture_output():
        bpe = SimpleBytePairEncoding.train(
            text, vocab_size=vocab_size, pat_str=gpt2_pattern
        )

    return bpe


def train_or_load_bpe(
    bpe_path: Path,
    text: str,
    vocab_size: int = 5000,
    preprocess: bool = False,
) -> SimpleBytePairEncoding:
    bpe_path = bpe_path / f"bpe_vocab={vocab_size}_preprocess={preprocess}.pkl"

    if preprocess:
        text = text.replace("\n", " ")
        text = text.lower()

    if bpe_path.exists():
        with bpe_path.open("rb") as f:
            return pickle.load(f)

    bpe = _train_bpe(text, vocab_size)

    with bpe_path.open("wb") as f:
        pickle.dump(bpe, f)

    return bpe
