import abc

__all__ = ["AbstractSimilarityMethod"]


class AbstractSimilarityMethod(abc.ABC):
    @abc.abstractmethod
    def match(self, text_1: str, text_2: str) -> float:
        pass
