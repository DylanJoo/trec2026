import random


def greedy_budget(doc_nuggets: dict, topk: int = 5) -> dict:
    """Deterministic setting. Select based on the gained nuggets."""
    selected = {}

    for qid in doc_nuggets:
        selected[qid] = []
        candidates = list(doc_nuggets[qid].keys())
        covered = set()

        for _ in range(topk):
            best_docid, best_gain = None, -1
            for docid in candidates:
                gain = len(set(doc_nuggets[qid][docid]) - covered)
                if gain > best_gain:
                    best_gain, best_docid = gain, docid

            if best_docid is None or best_gain == 0:
                break

            covered |= set(doc_nuggets[qid][best_docid])
            selected[qid].append(best_docid)
            candidates.remove(best_docid)

    return selected


def greedy_random(doc_nuggets: dict, topk: int = 5, seed: int = 42) -> dict:
    """Randomly sample topk docs per query."""
    rng = random.Random(seed)
    selected = {}
    for qid in doc_nuggets:
        pool = list(doc_nuggets[qid].keys())
        selected[qid] = rng.sample(pool, min(topk, len(pool)))
    return selected


def greedy_complete(doc_nuggets: dict) -> dict:
    """Greedily select docs per query until all nuggets are covered or no more gain."""
    selected = {}

    for qid in doc_nuggets:
        selected[qid] = []
        all_nuggets = {nid for nugids in doc_nuggets[qid].values() for nid in nugids}
        candidates = list(doc_nuggets[qid].keys())
        covered = set()

        while candidates and covered < all_nuggets:
            best_docid, best_gain = None, -1
            for docid in candidates:
                gain = len(set(doc_nuggets[qid][docid]) - covered)
                if gain > best_gain:
                    best_gain, best_docid = gain, docid

            if best_docid is None or best_gain == 0:
                break

            covered |= set(doc_nuggets[qid][best_docid])
            selected[qid].append(best_docid)
            candidates.remove(best_docid)

    return selected


# # -------------------------------------------------------------------------
# # Arrangement: re-order selected docids by nugget-coverage pattern
# # ---------------------------------------------------------------------------
#
# ARRANGEMENTS = ("rank", "desc_coverage", "asc_coverage", "random", "interleave")
#
#
# def arrange(selected: dict, doc_nuggets: dict, strategy: str = "rank", seed: int = 42) -> dict:
#     """
#     Re-order selected docids per query by nugget-coverage pattern.
#       rank          – preserve current order (no-op)
#       desc_coverage – most nuggets first
#       asc_coverage  – fewest nuggets first
#       random        – random shuffle
#       interleave    – alternate highest/lowest coverage
#     """
#     if strategy not in ARRANGEMENTS:
#         raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {ARRANGEMENTS}")
#
#     rng = random.Random(seed)
#     arranged = {}
#
#     for qid, docids in selected.items():
#         key = lambda d: len(doc_nuggets[qid].get(d, []))
#
#         if strategy == "rank":
#             arranged[qid] = list(docids)
#         elif strategy == "desc_coverage":
#             arranged[qid] = sorted(docids, key=key, reverse=True)
#         elif strategy == "asc_coverage":
#             arranged[qid] = sorted(docids, key=key)
#         elif strategy == "random":
#             out = list(docids)
#             rng.shuffle(out)
#             arranged[qid] = out
#         elif strategy == "interleave":
#             by_cov = sorted(docids, key=key, reverse=True)
#             lo, hi, out, turn = 0, len(by_cov) - 1, [], True
#             while lo <= hi:
#                 out.append(by_cov[lo] if turn else by_cov[hi])
#                 lo, hi = lo + turn, hi - (not turn)
#                 turn = not turn
#             arranged[qid] = out
#
#     return arranged
