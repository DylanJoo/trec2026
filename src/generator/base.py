from abc import ABC, abstractmethod
from src.data import Hit, Result


class BaseGenerator(ABC):

    @abstractmethod
    def generate(self, results: list[Result], contexts: dict[str, list[Hit]]) -> dict[str, str]:
        """
        Generate a response per query.
        contexts: {qid: [Hit, ...]} — the selected context documents.
        Returns {qid: response_text}.
        """
