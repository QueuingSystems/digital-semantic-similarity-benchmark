import collections.abc

__all__ = ["EmptyMapping"]


class EmptyMapping(collections.abc.MutableMapping):
    def __getitem__(self, key):
        raise KeyError(key)

    def __setitem__(self, key, val):
        pass

    def __delitem__(self, key):
        pass

    def __len__(self):
        return 0

    def __iter__(self):
        return []
