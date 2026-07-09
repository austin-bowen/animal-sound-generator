import random
from itertools import islice
from pathlib import Path
from socket import gethostname
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.functional import resample
from torchcodec.decoders import WavDecoder
from tqdm import tqdm

from asg.utils import doing, interleave

Split = Literal["train", "val", "test", "tiny"]

DEFAULT_SAMPLE_RATE = 22_050


@torch.no_grad()
def load_inatsounds(
    *,
    split: Split,
    species: str = "*_Animalia_Chordata_Mammalia_*",
    root: str | None = None,
    max_files: int | None = None,
    start_seconds: float = 1.0,
    min_seconds: float = 1.0,
    max_seconds: float | None = 2.0,
    resample_to: int = DEFAULT_SAMPLE_RATE,
    normalize_level: bool = True,
) -> np.ndarray:
    root: Path = _get_root(root)
    print(f"Loading {split} dataset from {root}")

    data_files = root.glob(f"{split}/{species}/*.wav")
    data_files = list(data_files)

    if max_files is not None:
        random.shuffle(data_files)
        data_files = islice(data_files, max_files)

    data = [
        _read_audio_file(
            path,
            start_seconds=start_seconds,
            # stop_seconds=None if max_seconds is None else start_seconds + max_seconds,
        )
        for path in tqdm(data_files, desc=f"Loading {split} dataset", unit="file")
    ]

    if max_seconds:
        max_len = round(max_seconds * DEFAULT_SAMPLE_RATE)
        data = interleave(_chunk_tensor(samples, max_len) for samples in data)

    # Exclude samples that are too short
    min_samples = min_seconds * DEFAULT_SAMPLE_RATE
    data = [samples for samples in data if samples.size(0) >= min_samples]

    # Find max length and pad all tensors to that length
    max_len = max(samples.size(0) for samples in data)
    data = [F.pad(samples, (0, max_len - samples.size(0))) for samples in data]

    data = torch.stack(data)

    if resample_to != DEFAULT_SAMPLE_RATE:
        with doing("Resampling"):
            data = resample(data, DEFAULT_SAMPLE_RATE, resample_to)

    if normalize_level:
        with doing("Normalizing"):
            data = data / data.abs().max(dim=1, keepdim=True).values

    return data.numpy()


def _get_root(root: str | None) -> Path:
    if not root:
        hostname = gethostname()
        if hostname == "austin-laptop":
            root = "./data/iNatSounds"
        elif hostname == "potato":
            root = "/mnt/data-fast/austin/datasets/iNatSounds"
        else:
            raise ValueError(
                f"Unknown hostname; must provide root. hostname={hostname}"
            )

    return Path(root)


def _chunk_tensor(tensor: torch.Tensor, max_len: int) -> tuple[torch.Tensor, ...]:
    """Chunk tensor into max_len-sized pieces, dropping the last chunk if too small."""

    num_chunks = tensor.size(0) // max_len
    if num_chunks == 0:
        return (tensor,)

    new_len = num_chunks * max_len
    tensor = tensor[:new_len]
    tensor = tensor.reshape(num_chunks, max_len)
    return tensor.unbind(0)


def _read_audio_file(
    path: Path,
    start_seconds: float = 0.0,
    stop_seconds: float | None = None,
) -> torch.Tensor:
    file = WavDecoder(path)

    try:
        samples = file.get_samples_played_in_range(
            start_seconds=start_seconds,
            stop_seconds=stop_seconds,
        )
    except RuntimeError as e:
        print(f"Failed to read {path}: {e!r}")
        return torch.zeros(1, dtype=torch.float32)

    return samples.data[0]


def main():
    data = load_inatsounds(
        split="tiny",
        max_seconds=5.0,
        resample_to=24_000,
    )

    print(data.shape)


if __name__ == "__main__":
    main()
