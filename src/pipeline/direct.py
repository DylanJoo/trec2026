"""Direct generation: no retrieval, empty context."""
from src.data import Result
from src.generator.base import BaseGenerator


class DirectPipeline:

    def __init__(self, generator: BaseGenerator):
        self.generator = generator

    def run(self, queries: dict[str, str]) -> dict[str, str]:
        results = [Result(qid=qid, query=query) for qid, query in queries.items()]
        contexts = {qid: [] for qid in queries}
        return self.generator.generate(results, contexts)
