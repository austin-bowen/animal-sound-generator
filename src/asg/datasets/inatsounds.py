from pathlib import Path
from socket import gethostname
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.functional import resample
from torchcodec.decoders import WavDecoder
from tqdm import tqdm

from asg.utils import doing

Split = Literal["train", "val", "test", "tiny"]

DEFAULT_SAMPLE_RATE = 22_050


def load_inatsounds(
    *,
    split: Split,
    root: str | None = None,
    max_seconds: float | None = None,
    resample_to: int = DEFAULT_SAMPLE_RATE,
) -> np.ndarray:
    root: Path = _get_root(root)

    data_files = root.glob(f"{split}/*/*.wav")
    data_files = list(data_files)

    data = [
        _read_audio_file(path)  # , stop_seconds=5.0)
        for path in tqdm(data_files, desc=f"Loading {split} dataset", unit="file")
    ]

    if max_seconds:
        max_len = round(max_seconds * DEFAULT_SAMPLE_RATE)

        data = [_chunk_tensor(row, max_len) for row in data]

        data = torch.concat(data, dim=0)
    else:
        # Find max length and pad all tensors to that length
        max_len = max(tensor.shape[0] for tensor in data)
        data = [F.pad(tensor, (0, max_len - tensor.shape[0])) for tensor in data]

        data = torch.stack(data)

    if resample_to != DEFAULT_SAMPLE_RATE:
        with doing("Resampling"):
            data = resample(data, DEFAULT_SAMPLE_RATE, resample_to)

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


def _chunk_tensor(tensor: torch.Tensor, max_len: int) -> torch.Tensor:
    """Chunk tensor into max_len-sized pieces, padding the last chunk if needed."""
    num_chunks = (tensor.shape[0] + max_len - 1) // max_len
    padded_len = num_chunks * max_len
    padded = F.pad(tensor, (0, padded_len - tensor.shape[0]))
    return padded.reshape(num_chunks, max_len)


def _read_audio_file(path: Path, stop_seconds: float | None = None) -> torch.Tensor:
    file = WavDecoder(path)
    samples = file.get_samples_played_in_range(stop_seconds=stop_seconds)
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
