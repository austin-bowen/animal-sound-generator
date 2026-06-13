from collections.abc import Iterable
from contextlib import contextmanager
from datetime import datetime
from itertools import chain, zip_longest
from typing import TypeVar

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
