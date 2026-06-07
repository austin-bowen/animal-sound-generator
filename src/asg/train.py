import argparse
import math
import os
from collections.abc import Callable
from typing import cast

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Dataset
from torch import nn
from torchaudio.functional import resample
from tqdm import tqdm

from asg.datasets.esc50 import load_animal_sounds_dataset
from asg.models.model0 import Model0
from asg.models.model1 import Model1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["esc50"],
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
) -> None:
    args = parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    if args.model == "model0":
        model = Model0()
    elif args.model == "model1":
        model = Model1()
    else:
        raise ValueError(f"Unknown model: {args.model}")
    model.to(device)

    def prepare_dataset(dataset_: Dataset) -> np.ndarray:
        dataset_ = np.stack(
            [
                row["audio"].get_all_samples().data
                for row in tqdm(dataset_, desc="Loading dataset")
            ]
        )
        dataset_ = dataset_.squeeze(1)
        dataset_ = torch.from_numpy(dataset_)
        dataset_ = resample(dataset_, 44_100, 24_000)
        return dataset_.numpy()

    dataset, test_dataset = load_animal_sounds_dataset(args.dataset)

    dataset = prepare_dataset(dataset)
    dataset = batch_apply_n2t2n(model.get_dac_z, dataset, device=device, batch_size=10)
    # dataset_std = np.std(dataset)
    dataset_std = 1
    dataset /= dataset_std
    # dataset = dataset[:80, :]
    print(f"train_dataset.shape={dataset.shape}")

    test_dataset = prepare_dataset(test_dataset)
    test_dataset /= dataset_std
    test_dataset = test_dataset[:40, :]

    samples = torch.from_numpy(test_dataset)
    save_audio("tmp/audio/dataset", samples, sample_rate=24_000)

    test_dataset = batch_apply_n2t2n(
        model.get_dac_z, test_dataset, device=device, batch_size=10
    )
    print(f"test_dataset.shape={test_dataset.shape}")

    optim = torch.optim.AdamW(
        model.parameters(),
        # lr=5e-4,
        # lr=2e-4,
        lr=1e-3,
        # weight_decay=1e-3,
        weight_decay=0,
    )

    if args.load:
        print(f"Loading checkpoint from {model_checkpoint_path}")

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

        loss_ = recon_loss_ + 0.0001 * kl_loss_
        # loss_ = recon_loss_ + corr_loss_
        # loss_ = cos_loss_ + 0.001 * mag_loss_

        return loss_, dict(
            recon_loss=recon_loss_.item(),
            kl_loss=kl_loss_.item(),
            # corr_loss=corr_loss_.item(),
            # cos_loss=cos_loss_.item(),
            # mag_loss=mag_loss_.item(),
        )

    batch_size = min(5, dataset.shape[0])
    for epoch in range(epoch_start, epoch_start + 1000):
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
