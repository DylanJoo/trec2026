from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Hit:
    docid: str
    score: float
    rank: int = -1
    content: str = ""


@dataclass
class Result:
    qid: str
    query: str
    hits: list[Hit] = field(default_factory=list)
    response: Optional[str] = None

    def sort(self):
        self.hits.sort(key=lambda h: h.score, reverse=True)
        for i, h in enumerate(self.hits):
            h.rank = i + 1

    def top_k(self, k: int) -> list[Hit]:
        return self.hits[:k]


def load_corpus(path: str) -> dict[str, str]:
    """Load a JSONL corpus: {docid, contents} per line."""
    import json
    corpus = {}
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc["docid"]] = doc.get("contents", doc.get("text", ""))
    return corpus


def load_queries(path: str) -> dict[str, str]:
    """Load a TSV queries file: qid<TAB>text."""
    queries = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
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


def load_nugget_qrels(path: str) -> dict[str, set[str]]:
    """Load nugget qrels for a single query: docid nugid relevance."""
    doc_nuggets: dict[str, set[str]] = defaultdict(set)
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3 and int(parts[2]) > 0:
                doc_nuggets[parts[0]].add(parts[1])
    return doc_nuggets
