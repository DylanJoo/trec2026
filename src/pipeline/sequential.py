"""One retriever → context selector → generator."""
from src.retriever.base import BaseRetriever
from src.generator.base import BaseGenerator


class SequentialPipeline:

    def __init__(self, retriever: BaseRetriever, selector, generator: BaseGenerator):
        self.retriever = retriever
        self.selector = selector
        self.generator = generator

    def run(self, queries: dict[str, dict], corpus: dict[str, dict]) -> dict[str, str]:
        results = self.retriever.retrieve(queries, corpus)
        contexts = {qid: self.selector.select(result) for qid, result in results.items()}
        return self.generator.generate(list(results.values()), contexts)
