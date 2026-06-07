from contextlib import contextmanager
from datetime import datetime


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
