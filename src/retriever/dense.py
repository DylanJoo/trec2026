import glob
import pickle
from contextlib import nullcontext
from itertools import chain

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.searcher import FaissFlatSearcher

from src.data import Hit, Result
from src.retriever.base import BaseRetriever


def _pickle_load(path):
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), lookup


class DenseRetriever(BaseRetriever):
    """On-the-fly dense retrieval: encodes queries and searches a pre-built FAISS index."""

    def __init__(
        self,
        index_pattern: str,
        model_name_or_path: str = "DylanJHJ/modernbert-base.cover-5k",
        query_prefix: str = "search_query: ",
        pooling: str = "mean",
        normalize: bool = True,
        query_max_len: int = 128,
        batch_size: int = 64,
        no_bf16: bool = False,
        append_eos: bool = False,
        topk: int = 100,
        quiet: bool = False,
    ):
        self.query_prefix = query_prefix
        self.query_max_len = query_max_len
        self.batch_size = batch_size
        self.append_eos = append_eos
        self.topk = topk
        self.quiet = quiet

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.float32 if no_bf16 else torch.bfloat16
        self._autocast = (
            torch.amp.autocast("cuda") if self.torch_dtype != torch.float32 else nullcontext()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = DenseModel.load(
            model_name_or_path,
            pooling=pooling,
            normalize=normalize,
            torch_dtype=self.torch_dtype,
            attn_implementation="flash_attention_2",
        ).to(self.device).eval()

        self._searcher, self._lookup = self._build_index(index_pattern)

    def _build_index(self, pattern: str):
        index_files = sorted(glob.glob(pattern))
        if not index_files:
            raise FileNotFoundError(f"No index files found: {pattern}")

        p_reps_0, p_lookup_0 = _pickle_load(index_files[0])
        searcher = FaissFlatSearcher(p_reps_0)

        shards = chain([(p_reps_0, p_lookup_0)], map(_pickle_load, index_files[1:]))
        if len(index_files) > 1:
            shards = tqdm(shards, desc="Loading index shards", total=len(index_files), disable=self.quiet)

        lookup = []
        for p_reps, p_lookup in shards:
            searcher.add(p_reps)
            lookup += p_lookup

        return searcher, lookup

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into query embeddings."""
        all_reps = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc="Encoding queries", disable=self.quiet):
            batch = texts[start: start + self.batch_size]
            max_len = self.query_max_len - 1 if self.append_eos else self.query_max_len
            encoded = self.tokenizer(
                batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
            )
            if self.append_eos:
                eos = self.tokenizer.eos_token_id
                encoded["input_ids"] = torch.cat(
                    [encoded["input_ids"], torch.full((len(batch), 1), eos, dtype=torch.long)], dim=1
                )
                encoded["attention_mask"] = torch.cat(
                    [encoded["attention_mask"], torch.ones((len(batch), 1), dtype=torch.long)], dim=1
                )
            batch_input = {k: v.to(self.device) for k, v in encoded.items()}
            with self._autocast, torch.no_grad():
                all_reps.append(self.model(query=batch_input).q_reps.cpu().float().numpy())
        return np.concatenate(all_reps, axis=0)

    def retrieve(
        self,
        queries: dict[str, dict],
        corpus: dict[str, dict] = None,
        topk: int = None,
    ) -> dict[str, Result]:
        topk = topk or self.topk
        corpus = corpus or {}
        qids = list(queries.keys())
        query_texts = [self.query_prefix + queries[qid]["query"] for qid in qids]

        q_reps = self.encode(query_texts)
        all_scores, all_indices = self._searcher.batch_search(
            q_reps, topk, self.batch_size, quiet=self.quiet
        )

        results = {}
        for i, qid in enumerate(qids):
            hits = []
            for rank, (idx, score) in enumerate(zip(all_indices[i], all_scores[i]), start=1):
                docid = str(self._lookup[idx])
                doc = corpus.get(docid, {})
                hits.append(Hit(
                    docid=docid,
                    score=float(score),
                    rank=rank,
                    content_dict={
                        "text": doc.get("text", "") if isinstance(doc, dict) else str(doc),
                        "title": doc.get("title", "") if isinstance(doc, dict) else "",
                    },
                ))
            results[qid] = Result(qid=qid, query=queries[qid]["query"], hits=hits, meta=queries[qid].get("meta", {}))
        return results
