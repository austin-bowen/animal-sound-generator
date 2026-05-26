import argparse

from asg.datasets import load_animal_sounds_dataset
from asg.models import Model0


def main() -> None:
    args = parse_args()
    print(args)

    dataset = load_animal_sounds_dataset(args.dataset)
    print(dataset)

    model = Model0()


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


if __name__ == "__main__":
    main()
