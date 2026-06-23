from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from itertools import chain, zip_longest
from typing import TypeVar

import matplotlib.pyplot as plt
from torch import nn

T = TypeVar("T")


@contextmanager
def doing(thing: str):
    print(f"{thing}...", end="", flush=True)

    start = datetime.now()
    try:
        yield
    except:
        print()
        raise

    dt = datetime.now() - start
    print(f" done ({dt})")


def graph_grads(model: nn.Module, out_path: str, bias: bool = True):
    """Graphs avg abs gradient by layer and saves to out_path."""

    names, values = [], []
    for name, param in model.named_parameters():
        if param.grad is not None and (bias or "bias" not in name):
            names.append(name)
            value = param.grad.abs().mean().item()
            values.append(value)

            if value == 0.0:
                print(f"WARN: Layer with zero grad: {name}")

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.4), 5))
    ax.bar(range(len(names)), values)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylabel("Mean Absolute Gradient")
    ax.set_title("Gradient Magnitudes by Layer")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def interleave(lst: Iterable[Iterable[T]]) -> Iterable[T]:
    """
    Interleaves the elements of the given list of lists.

    Example: [[1,2],[3,4,5],[6]] --> [1,3,6,2,4,5]
    """

    _sentinel = object()
    return (
        x
        for x in chain.from_iterable(zip_longest(*lst, fillvalue=_sentinel))
        if x is not _sentinel
    )


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad
