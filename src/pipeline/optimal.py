"""Optimal: load oracle docs from nugget qrels → greedy nugget selector → generator."""
from src.data import Hit, Result, get_oracle_docs
from src.generator.base import BaseGenerator


class OptimalPipeline:

    def __init__(self, selector, generator: BaseGenerator):
        self.selector = selector
        self.generator = generator

    def run(
        self,
        queries: dict[str, str],
        nugget_qrel_paths: dict[str, str],
        corpus: dict[str, str],
    ) -> dict[str, str]:
        """
        Args:
            queries: {qid: query_text}
            nugget_qrel_paths: {qid: path_to_nugget_qrel_file}
            corpus: {docid: text}
        """
        results = {}
        for qid, query in queries.items():
            qrel_path = nugget_qrel_paths[qid]
            oracle_docs = get_oracle_docs(qrel_path)
            hits = [
                Hit(docid=docid, score=1.0, rank=i + 1, content=corpus.get(docid, ""))
                for i, docid in enumerate(sorted(oracle_docs))
            ]
            results[qid] = Result(qid=qid, query=query, hits=hits)

        contexts = {qid: self.selector.select(result) for qid, result in results.items()}
        return self.generator.generate(list(results.values()), contexts)
