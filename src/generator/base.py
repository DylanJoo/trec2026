from abc import ABC, abstractmethod
from typing import Callable
from src.data import Hit, Result


class BaseGenerator(ABC):

    @abstractmethod
    def generate(
        self,
        results: list[Result],
        contexts: dict[str, list[Hit]],
        prompt_builder: Callable,
    ) -> dict[str, str]:
        """Returns {qid: response_text}."""
