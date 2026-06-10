from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Hit(dict):
    """Retrieval hit with attribute access and autollmrerank-compatible dict access."""

    def __init__(self, docid: str, score: float, rank: int = -1, content_dict: dict = None):
        super().__init__(docid=docid, score=score, rank=rank, content_dict=content_dict or {})

    @property
    def docid(self) -> str: return self["docid"]
    @property
    def score(self) -> float: return self["score"]
    @property
    def rank(self) -> int: return self["rank"]
    @property
    def content(self) -> str: return self["content_dict"]["text"]
    @property
    def title(self) -> str: return self["content_dict"]["title"]


@dataclass
class Result:
    qid: str
    query: str
    hits: list = field(default_factory=list)
    response: Optional[str] = None
    meta: dict = field(default_factory=dict)

    def sort(self):
        self.hits.sort(key=lambda h: h["score"], reverse=True)
        for i, h in enumerate(self.hits):
            h["rank"] = i + 1

    def sort_by(self, field: str = "score"):
        self.hits.sort(key=lambda x: x[field], reverse=True)
        for i, hit in enumerate(self.hits):
            hit["rank"] = i + 1

    def top_k(self, k: int) -> list:
        return self.hits[:k]

def truncate_hits(results: dict, max_model_len: int, n_docs: int, buffer: int = 100) -> None:
    """Truncate each hit's text in-place so all docs fit within max_model_len chars."""
    if n_docs <= 0 or max_model_len <= 0:
        return
    max_chars = max(0, max_model_len // n_docs - buffer) * 4
    for result in results.values():
        hits = result.hits if hasattr(result, "hits") else result
        for hit in hits:
            text = hit["content_dict"].get("text", "")
            if len(text) > max_chars:
                hit["content_dict"]["text"] = text[:max_chars]


def load_corpus(path: str, docids_filter=None) -> dict[str, dict]:
    from datasets import load_dataset
    ds = load_dataset('json', data_files=path, streaming=True)['train']

    corpus = {}
    for doc in ds:
        corpus[doc['id']] = {'title': doc['title'], 'content': doc['text']}

    return corpus

def load_queries(path: str) -> dict[str, dict]:
    """Load a unified JSONL queries file → {qid: {query, meta}}.
    Also accepts legacy TSV format: qid<TAB>query.
    """
    import json
    queries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                q = json.loads(line)
                queries[str(q["qid"])] = {"query": q["query"], "meta": q.get("meta", {})}
            else:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    queries[parts[0]] = {"query": parts[1], "meta": {}}
    return queries


def load_run(path: str, topk: int = 100) -> dict[str, list[tuple[str, float]]]:
    """Load a TREC run file: qid Q0 docid rank score tag."""
    run: dict[str, list] = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            qid, docid, rank, score = parts[0], parts[2], int(parts[3]), float(parts[4])
            if rank <= topk:
                run[qid].append((docid, score))
    # sort by score descending
    return {qid: sorted(hits, key=lambda x: x[1], reverse=True) for qid, hits in run.items()}


# TODO: think about the better qrel setting
def load_nugget_qrels(path: str) -> dict[str, dict[str, int]]:
    """Load nugget qrels: qid: docid: [nugid] --> nugget label is bianry.
    Return: 
        {qid: {docid: [1,2,3], docid: [1,4,5], ...} ...}
    """
    doc_nuggets = defaultdict(dict)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                qid, nugid, docid, rel = str(parts[0]), int(parts[1]), parts[2], int(parts[3])

                if qid not in doc_nuggets:
                    doc_nuggets[qid] = {}

                if docid not in doc_nuggets[qid]:
                    doc_nuggets[qid][docid] = []

                doc_nuggets[qid][docid].append(nugid)
    return doc_nuggets
