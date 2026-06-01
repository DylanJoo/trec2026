from src.data import Hit, Result, load_run
from src.retriever.base import BaseRetriever


class RunFileRetriever(BaseRetriever):
    """Loads a pre-computed TREC run file as retrieval results."""

    def __init__(self, run_path: str, topk: int = 100, name: str = "run"):
        self.run = load_run(run_path, topk=topk)
        self.topk = topk
        self.name = name

    def retrieve(self, queries: dict[str, str], corpus: dict[str, str]) -> dict[str, Result]:
        results = {}
        for qid, query in queries.items():
            hits = []
            for rank, (docid, score) in enumerate(self.run.get(qid, []), start=1):
                hits.append(Hit(docid=docid, score=score, rank=rank, content=corpus.get(docid, "")))
            results[qid] = Result(qid=qid, query=query, hits=hits)
        return results
