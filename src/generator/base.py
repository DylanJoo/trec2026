from abc import ABC, abstractmethod
from typing import Callable
from src.data import Result


class BaseGenerator(ABC):

    @abstractmethod
    def generate(
        self,
        results: dict[str, Result],
        prompt_builder: Callable,
    ) -> dict[str, str]:
        """Returns {qid: response_text}. Uses result.hits as context documents."""
