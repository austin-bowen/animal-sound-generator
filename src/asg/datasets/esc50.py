from typing import Literal

import numpy as np
import torch
from datasets import load_dataset
from torchaudio.functional import resample
from tqdm import tqdm

from asg.utils import doing

Split = Literal["train", "test"]

ESC_50_ANIMAL_CATEGORIES = (
    "cat",
    "cow",
    "crow",
    "dog",
    "frog",
    "hen",
    "insects",
    "pig",
    "rooster",
    "sheep",
)

ESC_50_SAMPLE_RATE = 44_100


def load_esc_50_animal_sounds(
    *,
    split: Split,
    resample_to: int = ESC_50_SAMPLE_RATE,
) -> np.ndarray:
    """
    Load the ESC-50 dataset filtered for animal sounds.

    Access a sound's tensor with: `ds[i]['audio'].get_all_samples().data`

    ```
    AudioSamples:
      data (shape): torch.Size([1, 220500])
      pts_seconds: 0.0
      duration_seconds: 5.0
      sample_rate: 44100
    ```
    """

    if split == "train":
        folds = {1, 2, 3, 4}
    elif split == "test":
        folds = {5}
    else:
        raise ValueError(f"Invalid split: {split}")

    ds = load_dataset("ashraq/esc50")

    ds = ds["train"].filter(
        lambda x: x["category"] in ESC_50_ANIMAL_CATEGORIES and x["fold"] in folds
    )

    ds = np.stack(
        [
            row["audio"].get_all_samples().data
            for row in tqdm(ds, desc=f"Loading {split} dataset")
        ]
    )
    ds = ds.squeeze(1)

    if resample_to != ESC_50_SAMPLE_RATE:
        with doing("Resampling"):
            ds = torch.from_numpy(ds)
            ds = resample(ds, ESC_50_SAMPLE_RATE, resample_to)
            ds = ds.numpy()

    return ds
