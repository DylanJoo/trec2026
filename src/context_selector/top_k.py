from src.data import Hit, Result


class TopKSelector:
    def __init__(self, k: int = 5):
        self.k = k

    def select(self, result: Result) -> list[Hit]:
        return result.hits[: self.k]
