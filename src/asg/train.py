import argparse
import math
import os
from collections.abc import Callable

import numpy as np
import torch
import torchaudio
from datasets import Dataset
from torch import nn
from torchaudio.functional import resample
from tqdm import tqdm

from asg.datasets import load_animal_sounds_dataset
from asg.models import Model0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        choices=["esc50"],
        required=True,
    )

    parser.add_argument(
        "--model",
        choices=["model0"],
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

    model = Model0()
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
    # dataset = dataset[:80, :]
    print(f"train_dataset.shape={dataset.shape}")

    test_dataset = prepare_dataset(test_dataset)
    test_dataset = test_dataset[:40, :]

    samples = torch.from_numpy(test_dataset)
    save_audio("tmp/audio/dataset", samples, sample_rate=24_000)

    test_dataset = batch_apply_n2t2n(model.get_dac_z, test_dataset, device=device, batch_size=10)
    print(f"test_dataset.shape={test_dataset.shape}")

    optim = torch.optim.AdamW(
        model.parameters(),
        # lr=5e-4,
        lr=2e-4,
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

    model.compile()

    recon_loss_fn = nn.MSELoss()

    corr_loss_mask = torch.tril(
        torch.ones(model.h_dim, model.h_dim, device=device),
        diagonal=-1,  # Exclude diagonal
    )

    def get_loss(z_hat_, z_, h_) -> tuple[torch.Tensor, dict[str, float]]:
        recon_loss_ = recon_loss_fn(z_hat_, z_)

        corr_loss_ = (h_.T @ h_) * corr_loss_mask
        corr_loss_ = corr_loss_.abs().mean()

        loss_ = recon_loss_ + corr_loss_

        return loss_, dict(
            recon_loss=recon_loss_.item(),
            corr_loss=corr_loss_.item(),
        )

    batch_size = min(5, dataset.shape[0])
    for epoch in range(epoch_start, epoch_start + 1000):
        # Shuffle dataset in place
        np.random.shuffle(dataset)

        model.train()
        losses = []
        for step in range(math.ceil(dataset.shape[0] / batch_size)):
            z = dataset[step * batch_size : (step + 1) * batch_size]
            z = torch.from_numpy(z).to(device)

            h, z_hat = model(z)

            loss, loss_dict = get_loss(z_hat, z, h)

            optim.zero_grad()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            loss.backward()
            optim.step()

            loss = loss.item()
            losses.append(loss)

            print(f"[e{epoch}; s{step}] loss: {loss:.4f} ({loss_dict})")

        avg_loss = np.mean(losses)
        print(f"[e{epoch}] avg loss: {avg_loss}")
        print()

        if epoch % 10 == 0:
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                z = torch.from_numpy(test_dataset).to(device)

                h, z_hat = model(z)

                loss, loss_dict = get_loss(z_hat, z, h)

                print(f"[e{epoch}] test loss: {loss:.4f} ({loss_dict})")
                print()

                samples = model.decode(h)
                save_audio(f"tmp/audio/epoch={epoch}", samples, sample_rate=24_000)

                del h, z_hat, z, samples
                torch.cuda.empty_cache()

            # Save state
            torch.save(
                dict(
                    model_state_dict=model.state_dict(),
                    optim_state_dict=optim.state_dict(),
                    epoch=epoch,
                ),
                model_checkpoint_path,
            )

            if avg_loss <= early_stop_loss:
                print('Early stop!')
                break


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
