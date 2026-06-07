import argparse
import math
import os
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn

from asg.datasets import load_esc_50_animal_sounds
from asg.datasets.inatsounds import load_inatsounds
from asg.models.model0 import Model0
from asg.models.model1 import Model1
from asg.utils import doing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=[
            "esc50",
            "inatsounds",
        ],
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=[
            "model0",
            "model1",
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
    early_stop_loss: float = 0.1,
    sample_rate: int = 24_000,
    samples_to_save: int = 40,
    epochs: int = 1000,
    batch_size: int = 10,
) -> None:
    args = parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    if args.dataset == "esc50":
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
            split="tiny",
            max_seconds=5.0,
            resample_to=sample_rate,
        )
        np.random.seed(42)
        np.random.shuffle(dataset)

        test_dataset = dataset[:40, :].copy()
        dataset = dataset[40:50, :]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    with doing(f"Saving {samples_to_save} test set samples"):
        samples = torch.from_numpy(test_dataset[:samples_to_save])
        save_audio("tmp/audio/dataset", samples, sample_rate=sample_rate)

    if args.model == "model0":
        model = Model0()
    elif args.model == "model1":
        model = Model1()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model.to(device)

    with doing("Converting training set to z"):
        dataset = batch_apply_n2t2n(
            model.get_dac_z,
            dataset,
            device=device,
            batch_size=batch_size,
        )

    with doing("Converting test set to z"):
        test_dataset = batch_apply_n2t2n(
            model.get_dac_z,
            test_dataset,
            device=device,
            batch_size=batch_size,
        )

    print(f"dataset.shape={dataset.shape}")
    print(f"test_dataset.shape={test_dataset.shape}")

    dac_z_std = dataset.std(axis=1).mean()
    print(f"dac_z_std={dac_z_std}")
    model.dac_z_std = dac_z_std
    dataset /= dac_z_std
    test_dataset /= dac_z_std

    optim = torch.optim.AdamW(
        model.parameters(),
        # lr=5e-4,
        # lr=2e-4,
        lr=2e-4,
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

    model.compile(dynamic=False)

    recon_loss_fn = nn.MSELoss()

    # corr_loss_mask = torch.tril(
    #     torch.ones(model.h_dim, model.h_dim, device=device),
    #     diagonal=-1,  # Exclude diagonal
    # )

    def get_loss(z_hat_, z_, mu_, log_var_) -> tuple[torch.Tensor, dict[str, float]]:
        recon_loss_ = recon_loss_fn(z_hat_, z_)
        kl_loss_ = -0.5 * (1 + log_var_ - mu_.pow(2) - log_var_.exp()).sum(dim=1).mean()

        # corr_loss_ = (mu_.T @ mu_) * corr_loss_mask
        # corr_loss_ = corr_loss_.abs().mean()

        # cos_loss_ = nn.functional.cosine_similarity(z_hat_, z_, dim=1)
        # cos_loss_ = 1 - cos_loss_.mean()

        # mag_loss_ = (z_hat_.norm(dim=1) - z_.norm(dim=1)).abs().mean()

        loss_ = recon_loss_ + 0.0 * kl_loss_
        # loss_ = recon_loss_ + corr_loss_
        # loss_ = cos_loss_ + 0.001 * mag_loss_

        return loss_, dict(
            recon_loss=recon_loss_.item(),
            kl_loss=kl_loss_.item(),
            # corr_loss=corr_loss_.item(),
            # cos_loss=cos_loss_.item(),
            # mag_loss=mag_loss_.item(),
        )

    for epoch in range(epoch_start, epoch_start + epochs):
        # Shuffle dataset in place
        np.random.shuffle(dataset)

        model.train()
        losses: dict[str, float] = run_epoch(
            model=model,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            loss_fn=get_loss,
            optim=optim,
        )
        avg_train_loss = losses["loss"]
        print(f"[e{epoch}] train loss: {losses}")

        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                losses, samples = run_epoch(
                    model=model,
                    dataset=test_dataset,
                    batch_size=batch_size,
                    device=device,
                    loss_fn=get_loss,
                    return_samples=True,
                )

                print(f"[e{epoch}]  test loss: {losses}")
                print()

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
    model: nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    device,
    loss_fn,
    optim: torch.optim.Optimizer | None = None,
    clip_grad: float | None = None,
    return_samples: bool = False,
) -> dict[str, float] | tuple[dict[str, float], torch.Tensor]:
    losses = []
    samples = []
    for step in range(math.ceil(dataset.shape[0] / batch_size)):
        z = dataset[step * batch_size : (step + 1) * batch_size]
        if z.shape[0] != batch_size:
            continue

        z = torch.from_numpy(z).to(device)

        mu, log_var, z_hat = model(z)

        loss, loss_dict = loss_fn(z_hat, z, mu, log_var)

        if optim:
            optim.zero_grad()

            loss.backward()
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optim.step()

        losses.append(
            {
                "loss": loss.item(),
                **loss_dict,
            }
        )

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
    main()
