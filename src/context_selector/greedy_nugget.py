from src.data import Hit, Result, load_nugget_qrels


class GreedyBudgetSelector:
    """
    Greedily select up to max_docs documents by marginal nugget gain.
    Stops early if no remaining document adds new nuggets.
    """

    def __init__(self, nugget_qrel_path: str, max_docs: int = 5):
        self.doc_nuggets = load_nugget_qrels(nugget_qrel_path)
        self.max_docs = max_docs

    def select(self, result: Result) -> list[Hit]:
        covered: set[str] = set()
        selected: list[Hit] = []
        candidates = list(result.hits)

        for _ in range(self.max_docs):
            best_hit, best_gain = None, -1
            for hit in candidates:
                gain = len(self.doc_nuggets.get(hit.docid, set()) - covered)
                if gain > best_gain:
                    best_gain, best_hit = gain, hit

            if best_hit is None or best_gain == 0:
                break

            covered |= self.doc_nuggets.get(best_hit.docid, set())
            selected.append(best_hit)
            candidates.remove(best_hit)

        return selected


class GreedyCompleteSelector:
    """
    Greedily select documents until all nuggets are covered (or no more gain).
    No fixed budget — stops at coverage saturation.
    """

    def __init__(self, nugget_qrel_path: str):
        self.doc_nuggets = load_nugget_qrels(nugget_qrel_path)

    def select(self, result: Result) -> list[Hit]:
        all_nuggets = set().union(*self.doc_nuggets.values()) if self.doc_nuggets else set()
        covered: set[str] = set()
        selected: list[Hit] = []
        candidates = list(result.hits)

        while candidates and covered < all_nuggets:
            best_hit, best_gain = None, -1
            for hit in candidates:
                gain = len(self.doc_nuggets.get(hit.docid, set()) - covered)
                if gain > best_gain:
                    best_gain, best_hit = gain, hit

            if best_hit is None or best_gain == 0:
                break

            covered |= self.doc_nuggets.get(best_hit.docid, set())
            selected.append(best_hit)
            candidates.remove(best_hit)

        return selected


class OracleAllSelector:
    """
    Include all oracle documents — upper bound on context.
    Documents are ordered by number of nuggets they cover (most informative first).
    """

    def __init__(self, nugget_qrel_path: str):
        self.doc_nuggets = load_nugget_qrels(nugget_qrel_path)

    def select(self, result: Result) -> list[Hit]:
        return sorted(
            result.hits,
            key=lambda h: len(self.doc_nuggets.get(h.docid, set())),
            reverse=True,
        )
