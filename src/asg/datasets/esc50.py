from datasets import Dataset, load_dataset

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


def load_animal_sounds_dataset(dataset: str) -> tuple[Dataset, Dataset]:
    if dataset == "esc50":
        return load_esc_50_animal_sounds()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def load_esc_50_animal_sounds() -> tuple[Dataset, Dataset]:
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

    ds = load_dataset("ashraq/esc50")

    ds = ds["train"].filter(lambda x: x["category"] in ESC_50_ANIMAL_CATEGORIES)

    train_folds = 4  # Out of 5
    train = ds.filter(lambda x: x["fold"] < train_folds)
    test = ds.filter(lambda x: x["fold"] >= train_folds)

    return train, test
