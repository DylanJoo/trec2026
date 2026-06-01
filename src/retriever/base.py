from abc import ABC, abstractmethod
from src.data import Result


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, queries: dict[str, str], corpus: dict[str, str]) -> dict[str, Result]:
        """Return a Result per query with hits populated."""
