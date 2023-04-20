from typing import Dict, List

from sklearn.model_selection import ParameterGrid

__all__ = ["merge_grids"]


def merge_grids(grids: List[Dict], _prev_params: Dict = None) -> List[Dict]:
    if len(grids) == 0:
        return []

    if _prev_params is None:
        _prev_params = {}

    for params in ParameterGrid(grids[0]):
        if len(grids) == 1:
            yield {**_prev_params, **params}
        else:
            yield from merge_grids(grids[1:], {**_prev_params, **params})


if __name__ == "__main__":
    grid = [
        [
            {"a": [True]},
            {"a": [False], "b": [1, 2]},
        ],
        {"c": [5, 6], "d": [7, 8]},
        {"e": ["foo", "far"], "f": [9, 10]},
    ]
    for params in merge_grids(grid):
        print(params)
