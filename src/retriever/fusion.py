from collections import defaultdict
from src.data import Hit, Result


def reciprocal_rank_fusion(
    results_list: list[dict[str, Result]],
    k: int = 60,
    topk: int = 100,
) -> dict[str, Result]:
    """RRF over multiple ranked lists. All lists must share the same query set."""
    all_qids = set()
    for results in results_list:
        all_qids.update(results.keys())

    fused: dict[str, Result] = {}
    for qid in all_qids:
        scores: dict[str, float] = defaultdict(float)
        query = ""
        corpus_lookup: dict[str, dict] = {}

        for results in results_list:
            if qid not in results:
                continue
            result = results[qid]
            query = result.query
            for hit in result.hits:
                scores[hit.docid] += 1.0 / (k + hit.rank)
                corpus_lookup[hit.docid] = hit["content_dict"]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        hits = [
            Hit(docid=docid, score=score, rank=i + 1, content_dict=corpus_lookup[docid])
            for i, (docid, score) in enumerate(ranked)
        ]
        fused[qid] = Result(qid=qid, query=query, hits=hits)

    return fused
