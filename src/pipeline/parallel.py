"""Multiple retrievers → RRF fusion → context selector → generator."""
from src.retriever.base import BaseRetriever
from src.retriever.fusion import reciprocal_rank_fusion
from src.generator.base import BaseGenerator


class ParallelPipeline:

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        selector,
        generator: BaseGenerator,
        rrf_k: int = 60,
    ):
        self.retrievers = retrievers
        self.selector = selector
        self.generator = generator
        self.rrf_k = rrf_k

    def run(self, queries: dict[str, str], corpus: dict[str, str]) -> dict[str, str]:
        all_results = [r.retrieve(queries, corpus) for r in self.retrievers]
        fused = reciprocal_rank_fusion(all_results, k=self.rrf_k)
        contexts = {qid: self.selector.select(result) for qid, result in fused.items()}
        return self.generator.generate(list(fused.values()), contexts)
