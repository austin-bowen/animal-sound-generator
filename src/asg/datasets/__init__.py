from datasets import Dataset

from asg.datasets.esc50 import load_esc_50_animal_sounds


def load_animal_sounds_dataset(dataset: str) -> tuple[Dataset, Dataset]:
    if dataset == "esc50":
        return load_esc_50_animal_sounds()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
