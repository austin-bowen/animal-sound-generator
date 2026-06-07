from contextlib import contextmanager


@contextmanager
def doing(thing: str):
    print(f"{thing}... ", end="", flush=True)
    try:
        yield
    except:
        print()
        raise
    print("done")
