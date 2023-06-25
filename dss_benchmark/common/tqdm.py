from tqdm import tqdm

__all__ = ["tqdm_v"]


def tqdm_v(it, verbose=True, **kwargs):
    if verbose:
        return tqdm(it, **kwargs)
    else:
        return it
