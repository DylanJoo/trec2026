from src.data import Hit, Result


class LLMReranker:
    """Wraps APRIL's AutoLLMReranker with trec2026's dict[str, Result] interface."""

    def __init__(self, method: str, model_name_or_path: str, depth: int = 100, **kwargs):
        from autollmrerank.wrapper import AutoLLMReranker
        kwargs.setdefault("top_k", depth)
        kwargs.setdefault("rank_end", depth)
        self.reranker = AutoLLMReranker.from_prebuilt(method, model_name_or_path, **kwargs)

    def rerank(
        self,
        results: dict[str, Result],
        query_batch_size: int = 32,
    ) -> dict[str, Result]:

        # Precache the contents
        content_cache = {}
        for result in results.values():
            for hit in result.hits:
                content_cache[hit['docid']] = hit['content_dict']

        # Rerank
        reranked_run = self.reranker.rerank(
            results=list(results.values()),
            query_batch_size=query_batch_size,
        )

        # Covert the run back to Result
        new_results: dict[str, Result] = {}
        for qid, docid_scores in reranked_run.items():
            orig = results[qid]
            ranked = sorted(docid_scores.items(), key=lambda x: x[1], reverse=True)

            # NOTE: here we have hit as a class. Maybe we should consider this in autollmrerank
            hits = []
            for rank, (docid, score) in enumerate(ranked, start=1):
                hits.append(Hit(
                        docid=docid, 
                        score=score,
                        rank=rank,
                        content_dict=content_cache[docid]
                ))
            new_results[qid] = Result(qid=qid, query=orig.query, hits=hits)

        return new_results
