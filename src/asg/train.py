import argparse
import math
import os
import random
from collections.abc import Callable
from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn

from asg.datasets import load_esc_50_animal_sounds
from asg.datasets.inatsounds import load_inatsounds
from asg.models.base import BaseDACModel
from asg.models.model1 import Model1
from asg.models.zelsa import ZELSA
from asg.models.zvae import ZVAE
from asg.utils import doing, graph_grads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=[
            "random",
            "esc50",
            "inatsounds",
        ],
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=[
            "model1",
            "zelsa",
            "zvae",
        ],
        required=True,
    )

    parser.add_argument(
        "--load",
        action="store_true",
        help="Load the model from disk.",
    )

    return parser.parse_args()


def main(
    model_checkpoint_path: str = "./tmp/model_checkpoint.pt",
    early_stop_loss: float = 0.001,
    sample_rate: int = 24_000,
    samples_to_save: int = 40,
    epochs: int = 1000,
    batch_size: int = 8,
    dtype: torch.dtype = torch.float32,
) -> None:
    args = parse_args()
    print(args)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    if args.dataset == "random":
        samples = round(sample_rate * 5.0)

        def gen_random_samples():
            s_ = np.random.random_sample((10 * batch_size, samples))
            s_ = s_ * 2 - 1
            return s_.astype(np.float32)

        dataset = gen_random_samples()
        test_dataset = gen_random_samples()
    elif args.dataset == "esc50":
        dataset = load_esc_50_animal_sounds(
            split="train",
            resample_to=sample_rate,
        )

        test_dataset = load_esc_50_animal_sounds(
            split="test",
            resample_to=sample_rate,
        )
    elif args.dataset == "inatsounds":
        dataset = load_inatsounds(
            split="train",
            max_files=1000,
            resample_to=sample_rate,
        )

        test_dataset = load_inatsounds(
            split="val",
            max_files=100,
            resample_to=sample_rate,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    model: BaseDACModel
    if args.model == "model1":
        model = Model1()
    elif args.model == "zelsa":
        model = ZELSA()
    elif args.model == "zvae":
        model = ZVAE()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model.to(device)

    with doing("Converting training set to z"):
        dataset = batch_apply_n2t2n(
            model.samples_to_z,
            dataset,
            device=device,
            batch_size=batch_size,
        )

    with doing("Converting test set to z"):
        test_dataset = batch_apply_n2t2n(
            model.samples_to_z,
            test_dataset,
            device=device,
            batch_size=batch_size,
        )

    print(f"dataset.shape={dataset.shape}")
    print(f"test_dataset.shape={test_dataset.shape}")

    dac_z_std = dataset.std(axis=2).mean()
    print(f"dac_z_std={dac_z_std}")
    model.dac_z_std = dac_z_std
    dataset /= dac_z_std
    test_dataset /= dac_z_std

    with doing(f"Saving {samples_to_save} test set samples"):
        z = test_dataset[:samples_to_save]
        z = torch.from_numpy(z).to(dtype=dtype, device=device)
        samples = model.z_to_samples(z)
        save_audio("tmp/audio/dataset", samples, sample_rate=sample_rate)

    optim = torch.optim.AdamW(
        model.parameters(),
        # lr=5e-4,
        # lr=2e-4,
        lr=1e-4,
        # weight_decay=1e-3,
        weight_decay=0,
    )

    if args.load:
        with doing(f"Loading checkpoint from {model_checkpoint_path}"):
            checkpoint = torch.load(
                model_checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            optim.load_state_dict(checkpoint["optim_state_dict"])
            epoch_start = checkpoint["epoch"] + 1
    else:
        epoch_start = 0

    model.to(dtype=dtype)
    model.compile(dynamic=False)

    for epoch in range(epoch_start, epoch_start + epochs):
        # Shuffle dataset in place
        np.random.shuffle(dataset)

        model.train()
        t0 = datetime.now()
        losses: dict[str, float] = run_epoch(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            optim=optim,
        )
        dt = datetime.now() - t0
        avg_train_loss = losses["loss"]
        print(f"[e{epoch}] train loss: {losses} dt={dt}")

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                losses, samples = run_epoch(
                    model=model,
                    dataset=test_dataset,
                    batch_size=batch_size,
                    dtype=dtype,
                    device=device,
                    return_samples=True,
                )

                print(f"[e{epoch}]  test loss: {losses}")
                print()

                samples = samples[:samples_to_save]
                save_audio(f"tmp/audio/epoch={epoch}", samples, sample_rate=24_000)

            # Save state
            torch.save(
                dict(
                    model_state_dict=model.state_dict(),
                    optim_state_dict=optim.state_dict(),
                    epoch=epoch,
                ),
                model_checkpoint_path,
            )

            if avg_train_loss <= early_stop_loss:
                print("Early stop!")
                break


def run_epoch(
    *,
    model: BaseDACModel,
    dataset: np.ndarray,
    batch_size: int,
    dtype: torch.dtype,
    device,
    optim: torch.optim.Optimizer | None = None,
    norm_grad: bool = False,
    clip_grad: float | None = None,
    return_samples: bool = False,
) -> dict[str, float] | tuple[dict[str, float], torch.Tensor]:
    losses = []
    samples = []
    for step in range(math.ceil(dataset.shape[0] / batch_size)):
        z = dataset[step * batch_size : (step + 1) * batch_size]
        if z.shape[0] != batch_size:
            continue

        z = torch.from_numpy(z).to(dtype=dtype, device=device)

        z_hat, extras = model(z)

        loss, loss_dict = model.get_loss(z, z_hat, extras)

        if optim:
            optim.zero_grad()

            loss.backward()

            # Normalize gradients
            if norm_grad:
                for param in model.parameters():
                    if param.grad is None:
                        continue
                    param.grad = param.grad / param.grad.norm()

            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optim.step()

            if step == 0:
                graph_grads(model, "tmp/grads.png")
                # input("Press Enter to continue...")

        losses.append(loss_dict)

        if return_samples:
            batch_samples = model.z_to_samples(z_hat)
            batch_samples = batch_samples.detach().cpu()
            samples.append(batch_samples)

    losses = pd.DataFrame(losses).mean().to_dict()
    losses = cast(dict[str, float], losses)

    if return_samples:
        samples = torch.concat(samples, dim=0)
        return losses, samples
    else:
        return losses


def batch_apply_n2t2n(
    fn: Callable[[torch.Tensor], torch.Tensor],
    data: np.ndarray,
    device=None,
    batch_size: int = 1,
) -> np.ndarray:
    results = []

    batch_count = math.ceil(data.shape[0] / batch_size)
    for i in range(batch_count):
        batch = data[i * batch_size : (i + 1) * batch_size]
        batch = torch.from_numpy(batch).to(device)
        results.append(fn(batch).detach().cpu().numpy())

    results = np.concatenate(results, axis=0)

    assert results.shape[0] == data.shape[0]
    return results


def save_audio(path, batch: torch.Tensor, sample_rate: int) -> None:
    os.makedirs(path, exist_ok=True)
    batch = batch.cpu()
    for row in range(batch.size(0)):
        audio = batch[row : row + 1]
        torchaudio.save(f"{path}/{row}.wav", audio, sample_rate)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
