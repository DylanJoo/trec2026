from src.data import Hit, Result, load_run
from src.retriever.base import BaseRetriever


class RunFileRetriever(BaseRetriever):
    """Loads a pre-computed TREC run file as retrieval results."""

    def __init__(self, run_path: str, topk: int = 100, name: str = "run"):
        self.run = load_run(run_path, topk=topk)
        self.topk = topk
        self.name = name

    def retrieve(self, queries: dict[str, dict], corpus: dict[str, dict]) -> dict[str, Result]:
        results = {}
        for qid, q in queries.items():
            hits = []
            for rank, (docid, score) in enumerate(self.run.get(qid, []), start=1):
                doc = corpus.get(docid, {})
                hits.append(Hit(
                    docid=docid,
                    score=score,
                    rank=rank,
                    content_dict={
                        "text": doc.get("text", "") if isinstance(doc, dict) else doc,
                        "title": doc.get("title", "") if isinstance(doc, dict) else "",
                    },
                ))
            results[qid] = Result(qid=qid, query=q["query"], hits=hits)
        return results
