import argparse
import glob
import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from tevatron.retriever.modeling import DenseModel
from tevatron.retriever.searcher import FaissFlatSearcher

from src.data import load_queries

# DATASETS = [
#     {
#         "name": "neuclir2024",
#         "queries": "data/neuclir2024.topics.test.jsonl",
#         "index_dir": "/home/dju/scratch/neuclir1/{model}/corpus_emb*pkl",
#         "run_output": "runs/runs.neuclir2024.{tag}.test.txt",
#     },
#     {
#         "name": "ragtime2025",
#         "queries": "data/ragtime2025.topics.test.jsonl",
#         "index_dir": "/home/dju/scratch/ragtime1/{model}/corpus_emb*pkl",
#         "run_output": "runs/runs.ragtime2025.{tag}.test.txt",
#     },
#     {
#         "name": "ragtime2026",
#         "queries": "data/ragtime2026.topics.test.jsonl",
#         "index_dir": "/home/dju/scratch/ragtime2/{model}/corpus_emb*pkl",
#         "run_output": "runs/runs.ragtime2026.{tag}.test.txt",
#     },
# ]

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="DylanJHJ/modernbert-base.cover-5k")
parser.add_argument("--query_prefix", default="Search Query: ")
parser.add_argument("--pooling", default="eos")
parser.add_argument("--normalize", action="store_true", default=True)
parser.add_argument("--lora", default=None)
parser.add_argument("--query_max_len", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--append_eos", action="store_true", default=True)
parser.add_argument("--no_bf16", action="store_true")
parser.add_argument("--depth", type=int, default=1000)
parser.add_argument("--search_batch", type=int, default=128)
args = parser.parse_args()

model_shortname = args.model.split("/")[-1]
datasets = [d for d in DATASETS if args.datasets is None or d["name"] in args.datasets]

# Load model once, reuse across datasets
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float32 if args.no_bf16 else torch.bfloat16

print(f"Loading model {args.model} on {device}")
tokenizer = AutoTokenizer.from_pretrained(args.model)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = DenseModel.load(
    args.model,
    pooling=args.pooling,
    normalize=args.normalize,
    lora_name_or_path=args.lora,
    torch_dtype=torch_dtype,
    attn_implementation="flash_attention_2",
)
model = model.to(device).eval()
autocast_ctx = torch.amp.autocast("cuda") if torch_dtype != torch.float32 else nullcontext()


def encode_queries(query_texts):
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
    return np.concatenate(all_q_reps, axis=0)


def load_shard(path):
    with open(path, "rb") as f:
        reps, lookup = pickle.load(f)
    return np.array(reps), list(lookup)


def build_index(index_pattern):
    index_files = sorted(glob.glob(index_pattern))
    if not index_files:
        raise FileNotFoundError(f"No corpus embedding files: {index_pattern}")

    p_reps, p_lookup = load_shard(index_files[0])
    searcher = FaissFlatSearcher(p_reps)
    lookup = p_lookup

    for path in tqdm(index_files[1:], desc="Loading corpus shards"):
        p_reps, p_lookup = load_shard(path)
        searcher.add(p_reps)
        lookup += p_lookup

    try:
        import faiss
        num_gpus = faiss.get_num_gpus()
        if num_gpus == 1:
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            searcher.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, searcher.index, co)
        elif num_gpus > 1:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = co.useFloat16 = True
            searcher.index = faiss.index_cpu_to_all_gpus(searcher.index, co, ngpu=num_gpus)
    except Exception:
        pass

    return searcher, lookup


for ds in datasets:
    print(f"\n=== {ds['name']} ===")

    queries = load_queries(ds["queries"])
    qids = list(queries.keys())
    query_texts = [args.query_prefix + queries[qid]["query"] for qid in qids]

    q_reps = encode_queries(query_texts)

    index_pattern = ds["index_dir"].format(model=model_shortname)
    searcher, lookup = build_index(index_pattern)

    all_scores, all_indices = searcher.batch_search(q_reps, args.depth, args.search_batch, quiet=False)

    run_output = ds["run_output"].format(tag=args.tag)
    os.makedirs(os.path.dirname(run_output) or ".", exist_ok=True)
    with open(run_output, "w") as f:
        for qid, scores, indices in zip(qids, all_scores, all_indices):
            for rank, (score, idx) in enumerate(sorted(zip(scores, indices), reverse=True), 1):
                f.write(f"{qid} Q0 {lookup[idx]} {rank} {score:.6f} {args.tag}\n")

    print(f"Wrote {len(qids)} queries → {run_output}")
