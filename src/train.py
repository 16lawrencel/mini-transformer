import torch
from torch import nn
from torch.utils.data import DataLoader

from pathlib import Path
import os
from typing import Literal

total_times = []


def _get_checkpoint_filename(
    checkpoint_file: Path,
    epoch: int,
) -> str:
    return str(checkpoint_file / f"epoch_{epoch}.pt")


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_epochs: int,
    checkpoint_file: Path,
    vocab_size: int,
    device: Literal["cpu", "mps", "cuda"],
    starting_epoch: int = 0,
) -> list[float]:
    SENTINEL = vocab_size - 1

    if not os.path.exists(checkpoint_file):
        os.makedirs(checkpoint_file)

    if starting_epoch != 0:
        print(f"Loading model weights for starting epoch {starting_epoch}...")
        model.load_state_dict(
            torch.load(
                _get_checkpoint_filename(checkpoint_file, starting_epoch),
                map_location=device,
            )
        )
        print("Finished loading weights!")

    model.train()
    model = model.to(device)

    losses = []

    for epoch in range(starting_epoch + 1, starting_epoch + num_epochs + 1):
        total_loss = 0
        total_count = 0
        for i, data in enumerate(dataloader):
            # print(f"[Epoch {epoch}] i = {i}")
            data = data.to(device)

            # x is tokens 0..n-2
            # try to predict tokens 1..n-1
            x = data[:, :-1]
            expected = data[:, 1:]

            optimizer.zero_grad()
            preds = model(x)
            preds_flatten = preds.reshape(-1, vocab_size)
            expected_flatten = expected.flatten()

            # print("Expected flatten: ", expected_flatten)
            loss = loss_fn(preds_flatten, expected_flatten.long())

            total_loss += loss.to("cpu") * len(x)
            total_count += len(x)

            loss.backward()
            optimizer.step()

        scheduler.step()

        avg_loss = total_loss / total_count
        print(f"[Epoch {epoch}] avg loss = {avg_loss:.4f}")
        losses.append(avg_loss)

        if epoch % 100 == 0:
            torch.save(
                model.state_dict(), _get_checkpoint_filename(checkpoint_file, epoch)
            )

    return losses
