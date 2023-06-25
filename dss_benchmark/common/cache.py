import collections.abc
import pickle
from typing import Union

import cachetools
import redis

__all__ = ["EmptyMapping", "RedisCache", "init_cache"]


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


class RedisCache(collections.abc.MutableMapping):
    def __init__(
        self,
        *,
        namespace: str = "redis_cache:",
        client: redis.Redis = None,
    ):
        self.__namespace = namespace
        self.client = client or redis.Redis()
        if client.ping() is not True:
            raise Exception("Cannot ping Redis")

    def __getitem__(self, key: str):
        try:
            val = self.client.get(self.__namespace + str(key))
            if val:
                return self.deserialize(val)
            else:
                raise KeyError()
        except KeyError:
            return self.__missing__(str(key))

    def __setitem__(self, key: str, value):
        self.client.set(self.__namespace + str(key), self.serialize(value))

    def __delitem__(self, key: str):
        self.client.delete(self.__namespace + str(key))

    def __contains__(self, key: str):
        return self.client.exists(self.__namespace + str(key)) == 1

    def __missing__(self, key: str):
        raise KeyError(key)

    def __iter__(self):
        for key in self.client.scan_iter(match=self.__namespace + "*"):
            yield key.decode("utf-8")[len(self.__namespace) :]

    def __len__(self):
        return len(self.client.keys(pattern=self.__namespace + "*"))

    def clear(self):
        for key in self:
            try:
                del self[key]
            except:
                pass

    @staticmethod
    def deserialize(value) -> bytes:
        return pickle.loads(value)

    @staticmethod
    def serialize(value) -> bytes:
        return pickle.dumps(value)


def init_cache(kind: Union[str, None] = None):
    if kind == "redis":
        return RedisCache(client=redis.Redis(host="localhost", password="12345"))
    elif kind == "memory":
        return cachetools.LFUCache(2**16)
    elif kind == 'dummy':
        return EmptyMapping()
    try:
        cache = RedisCache(client=redis.Redis(host="localhost", password="12345"))
    except Exception:
        print("Cannot initialize redis")
        cache = cachetools.LFUCache(2**16)
    return cache
