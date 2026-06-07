"""Oracle generation: context built from ground-truth relevant docs in nugget qrels.

Example usage:
  python run_oracle.py config/ragtime.yaml
"""
import json
import os
import sys
import torch

import yaml

from src.data import load_queries, load_corpus, Result, Hit, load_nugget_qrels, truncate_hits
from src.generator import VLLMGenerator, EndpointGenerator
from src.generator.prompt import get_prompt_builder
from src.context_selector.greedy_nugget import greedy_budget

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg["data"]
gen_cfg = cfg["generator"]
sel_cfg = cfg["selector"]
exp_cfg = cfg["exp"]

exp_name = exp_cfg["exp_name"]
context_path = exp_cfg["context_path"].format(exp_name=exp_name)
output_path = exp_cfg["response_path"].format(exp_name=exp_name)
mode = gen_cfg.get("mode", "vllm")

# Load data
queries = load_queries(data_cfg["queries_path"])
corpus = load_corpus(data_cfg["corpus_path"])
doc_nuggets = load_nugget_qrels(data_cfg["nugget_qrels_path"])

# Oracle context selection
if sel_cfg["topk"] == 0:
    selected = greedy_complete(doc_nuggets)
else:
    selected = greedy_budget(doc_nuggets, topk=sel_cfg["topk"])

# Build result list and contexts from selected docs
result_list = []
contexts = {}

for qid, q in queries.items():
    docids = selected.get(qid, [])
    hits = []
    for rank, docid in enumerate(docids, start=1):
        doc = corpus.get(docid)
        hits.append(Hit(
            docid=docid,
            score=0.0,
            rank=rank,
            content_dict={"text": doc["content"], "title": doc.get("title", "")},
        ))

    result_list.append(Result(qid=qid, query=q["query"], meta=q.get("meta", {})))
    contexts[qid] = hits

# Truncate document text to fit within model context
max_model_len = gen_cfg.get("max_model_len", 10240)
truncate_hits(contexts, max_model_len=max_model_len, n_docs=sel_cfg["topk"])

# Generation
if mode == "endpoint":
    generator = EndpointGenerator(
        gen_cfg["model_name_or_path"],
        temperature=gen_cfg.get("temperature", 0.0),
        top_p=gen_cfg.get("top_p", 1.0),
        max_tokens=gen_cfg.get("max_tokens", 512),
        base_url=gen_cfg.get("base_url", "http://localhost:8000/v1"),
        api_key=gen_cfg.get("api_key", "EMPTY"),
    )
else:
    generator = VLLMGenerator(
        gen_cfg["model_name_or_path"],
        temperature=gen_cfg.get("temperature", 0.0),
        top_p=gen_cfg.get("top_p", 1.0),
        max_tokens=gen_cfg.get("max_tokens", 512),
        dtype=gen_cfg.get("dtype", "bfloat16"),
        gpu_memory_utilization=gen_cfg.get("gpu_memory_utilization", 0.9),
        num_gpus=torch.cuda.device_count(),
        max_model_len=gen_cfg.get("max_model_len", 10240),
    )

prompt_builder = get_prompt_builder(gen_cfg.get("track", "rag"))
responses = generator.generate(result_list, contexts, prompt_builder=prompt_builder)

# Write responses
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
with open(output_path, "w") as f:
    for qid, text in responses.items():
        f.write(json.dumps({"qid": qid, "response": text}) + "\n")
print(f"Wrote {len(responses)} responses to {output_path}")

# Write TREC context run
os.makedirs(os.path.dirname(context_path) or ".", exist_ok=True)
with open(context_path, "w") as f:
    for qid, hits in contexts.items():
        for hit in hits:
            f.write(f"{qid} Q0 {hit.docid} {hit.rank} {hit.score} oracle\n")
print(f"Wrote TREC run to {context_path}")
