from src.data import Hit, Result, load_nugget_qrels


class GreedyNuggetSelector:
    """
    Greedily select documents that maximize nugget coverage.
    Requires a per-query nugget qrel file: docid nugid relevance.
    """

    def __init__(self, nugget_qrel_path: str, max_docs: int = 5):
        self.doc_nuggets = load_nugget_qrels(nugget_qrel_path)
        self.max_docs = max_docs

    def select(self, result: Result) -> list[Hit]:
        covered: set[str] = set()
        selected: list[Hit] = []

        for _ in range(self.max_docs):
            best_hit = None
            best_gain = -1

            for hit in result.hits:
                if any(h.docid == hit.docid for h in selected):
                    continue
                gain = len(self.doc_nuggets.get(hit.docid, set()) - covered)
                if gain > best_gain:
                    best_gain = gain
                    best_hit = hit

            if best_hit is None or best_gain == 0:
                break

            covered |= self.doc_nuggets.get(best_hit.docid, set())
            selected.append(best_hit)

        return selected
