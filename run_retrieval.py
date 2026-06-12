import argparse
import glob
import os
import pickle
import sys
from contextlib import nullcontext

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.searcher import FaissFlatSearcher

from tevatron.retriever.driver.search import (
    pickle_load, 
    pickle_save,
    search_queries,
    write_ranking,
)

from src.data import load_queries
from itertools import chain

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passage_reps", required=True)
    parser.add_argument("--model_name_or_path", default="DylanJHJ/modernbert-base.cover-5k")
    parser.add_argument("--queries", required=True)
    parser.add_argument("--query_prefix", default="search_query: ")
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--query_max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--depth", type=int, default=1000)
    parser.add_argument("--append_eos", action="store_true")
    # unused
    parser.add_argument("--output", dest="save_ranking_to", required=True)
    parser.add_argument("--save_text", action="store_true")
    parser.add_argument('--aggregation_strategy', type=str, default='mean') 
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    # Load queries
    queries = load_queries(args.queries)
    qids = list(queries.keys())
    query_texts = [args.query_prefix + queries[qid]["query"] for qid in qids]

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float32 if args.no_bf16 else torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = DenseModel.load(
        args.model_name_or_path,
        pooling=args.pooling,
        normalize=args.normalize,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device).eval()

    # Encode queries
    autocast_ctx = torch.amp.autocast("cuda") if torch_dtype != torch.float32 else nullcontext()
    all_q_reps = []

    for start in tqdm(range(0, len(query_texts), args.batch_size), desc="Encoding queries"):
        batch = query_texts[start: start + args.batch_size]
        max_len = args.query_max_len - 1 if args.append_eos else args.query_max_len
        encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        if args.append_eos:
            eos = tokenizer.eos_token_id
            encoded["input_ids"] = torch.cat(
                [encoded["input_ids"], torch.full((len(batch), 1), eos, dtype=torch.long)], dim=1
            )
            encoded["attention_mask"] = torch.cat(
                [encoded["attention_mask"], torch.ones((len(batch), 1), dtype=torch.long)], dim=1
            )
        batch_input = {k: v.to(device) for k, v in encoded.items()}
        with autocast_ctx, torch.no_grad():
            all_q_reps.append(model(query=batch_input).q_reps.cpu().float().numpy())

    q_reps = np.concatenate(all_q_reps, axis=0)

    # Build FAISS index from corpus shards
    """The pipeline below is inherited from the original Tevatron codebase:
    reference: tevatron/retriever/driver/search.py
    """
    index_files = sorted(glob.glob(args.passage_reps))
    if not index_files:
        raise FileNotFoundError(f"No corpus embedding files: {args.passage_reps}")
    p_reps_0, p_lookup_0 = pickle_load(index_files[0])
    retriever = FaissFlatSearcher(p_reps_0)

    shards = chain([(p_reps_0, p_lookup_0)], map(pickle_load, index_files[1:]))
    if len(index_files) > 1:
        shards = tqdm(shards, desc="Loading shards into index", total=len(index_files))
    look_up = []
    for p_reps, p_lookup in shards:
        retriever.add(p_reps)
        look_up += p_lookup

    num_gpus = faiss.get_num_gpus()
    all_scores, psg_indices = search_queries(retriever, q_reps, look_up, args)

    # Search and write TREC run
    os.makedirs(os.path.dirname(args.save_ranking_to), exist_ok=True) if os.path.dirname(args.save_ranking_to) else None
    write_ranking(psg_indices, all_scores, qids, args.save_ranking_to, trec_format=True, trec_run_name='tevatron')

if __name__ == "__main__":
    main()
