from typing import List

import numpy as np

__all__ = ["limit_array"]


def limit_array(array: List[int], limit: int):
    sum = np.sum(array)
    sorted_idx = np.argsort(-np.array(array))
    while sum > limit:
        idx = -1
        for i, i_1 in zip(sorted_idx, sorted_idx[1:]):
            if array[i] >= array[i_1]:
                idx = i
                break
        sum -= 1
        array[idx] -= 1
    return array


if __name__ == "__main__":
    print(limit_array([10, 9, 1, 2], 10))
    print(limit_array([10, 9, 1, 2], 15))
    print(limit_array([0, 10, 2, 13, 1], 15))
