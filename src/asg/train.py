import argparse
import os

import numpy as np
import torch
import torchaudio
from torch import nn
from torchaudio.functional import resample
from tqdm import tqdm

from asg.datasets import load_animal_sounds_dataset
from asg.models import Model0

SAVE_SAMPLE_COUNT = 40


def main() -> None:
    args = parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    dataset = load_animal_sounds_dataset(args.dataset)

    dataset = np.stack(
        [
            row["audio"].get_all_samples().data
            for row in tqdm(dataset, desc="Loading dataset")
        ]
    )
    dataset = dataset.squeeze(1)
    dataset = torch.from_numpy(dataset)
    dataset = resample(dataset, 44_100, 24_000).numpy()
    dataset = dataset[:80, :]
    print(f"dataset.shape={dataset.shape}")

    test_dataset = dataset[:SAVE_SAMPLE_COUNT].copy()
    samples = torch.from_numpy(test_dataset)
    save_audio("tmp/audio/dataset", samples, sample_rate=24_000)

    model = Model0()
    model.to(device)
    model.compile()

    # with torch.no_grad():
    #     _, dataset_z = model.encode(torch.from_numpy(dataset[:100]).to(device))
    #     dataset_z = dataset_z.detach().cpu().numpy()

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=5e-4,
        # weight_decay=1e-3,
        weight_decay=0,
    )

    recon_loss_fn = nn.MSELoss()

    batch_size = min(5, dataset.shape[0])
    for epoch in range(1000):
        # Shuffle dataset in place
        np.random.shuffle(dataset)

        model.train()
        losses = []
        for step in range(dataset.shape[0] // batch_size):
            samples = dataset[step * batch_size : (step + 1) * batch_size]
            # samples = dataset_z[step * batch_size : (step + 1) * batch_size]
            samples = torch.from_numpy(samples).to(device)

            h, z_hat, z = model(samples)
            # z = samples
            # h = model.encoder(z)
            # z_hat = model.decoder(h)

            recon_loss = recon_loss_fn(z_hat, z)
            # corr_loss = (
            #     ((h.T @ h) * (1 - torch.eye(h.shape[1], device=device))).pow(2).mean()
            # )
            corr_loss = torch.tensor(0.0)
            loss = recon_loss# + corr_loss

            optim.zero_grad()
            # nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            loss.backward()
            optim.step()

            loss = loss.item()
            losses.append(loss)
            print(
                f"[e{epoch}; s{step}] loss: {loss} "
                f"(recon: {recon_loss.item()}, corr: {corr_loss.item()})"
            )

        avg_loss = np.mean(losses)
        print(f"[e{epoch}] avg loss: {avg_loss}")
        print()

        if epoch % 10 == 0:
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                samples = torch.from_numpy(test_dataset).to(device)
                samples = model.decode(model.encode(samples)[0])
                # samples = model.dac_model.decode(model.encode(samples)[1]).squeeze(1)
                save_audio(f"tmp/audio/epoch={epoch}", samples, sample_rate=24_000)

                del samples
                torch.cuda.empty_cache()

            if avg_loss < 1.0:
                print('Early stop!')
                break


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

    return parser.parse_args()


def save_audio(path, batch: torch.Tensor, sample_rate: int) -> None:
    os.makedirs(path, exist_ok=True)
    batch = batch.cpu()
    for row in range(batch.size(0)):
        audio = batch[row : row + 1]
        torchaudio.save(f"{path}/{row}.wav", audio, sample_rate)


if __name__ == "__main__":
    main()
