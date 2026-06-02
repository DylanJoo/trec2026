"""Optimal: load oracle docs from nugget qrels → context selector → generator."""
from src.data import Hit, Result, get_oracle_docs
from src.generator.base import BaseGenerator


class OptimalPipeline:

    def __init__(self, selector_factory, generator: BaseGenerator):
        """
        selector_factory: callable(nugget_qrel_path) → selector with .select(result)
        This is needed because each query has its own nugget qrel file.
        """
        self.selector_factory = selector_factory
        self.generator = generator

    def run(
        self,
        queries: dict[str, str],
        nugget_qrel_paths: dict[str, str],
        corpus: dict[str, str],
    ) -> dict[str, str]:
        results = {}
        contexts = {}

        for qid, query in queries.items():
            qrel_path = nugget_qrel_paths[qid]
            oracle_docs = get_oracle_docs(qrel_path)
            hits = [
                Hit(docid=docid, score=1.0, rank=i + 1, content=corpus.get(docid, ""))
                for i, docid in enumerate(sorted(oracle_docs))
            ]
            result = Result(qid=qid, query=query, hits=hits)
            results[qid] = result

            selector = self.selector_factory(qrel_path)
            contexts[qid] = selector.select(result)

        return self.generator.generate(list(results.values()), contexts)
