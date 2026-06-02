"""Direct generation: no retrieval, empty context."""
from src.data import Result
from src.generator.base import BaseGenerator


class DirectPipeline:

    def __init__(self, generator: BaseGenerator):
        self.generator = generator

    def run(self, queries: dict[str, dict]) -> dict[str, str]:
        results = [
            Result(qid=qid, query=q["query"], meta=q.get("meta", {}))
            for qid, q in queries.items()
        ]
        contexts = {r.qid: [] for r in results}
        return self.generator.generate(results, contexts)
