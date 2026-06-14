"""RAG pipeline: retrieval + generation.

Example usage:
  python run_rag.py config/ragtime.yaml
"""
import json
import os
import sys

import torch
import yaml

from src.data import load_queries, load_corpus, truncate_hits
from src.retriever.run_file import RunFileRetriever
from src.context_selector.trivial import select_top_k
from src.generator import VLLMGenerator, EndpointGenerator
from src.generator.prompt import get_prompt_builder

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg["data"]
gen_cfg = cfg["generator"]
retriever_cfg = cfg["retriever"]
sel_cfg = cfg["selector"]
exp_cfg = cfg["exp"]

exp_name = exp_cfg["exp_name"]
context_path = exp_cfg["context_path"].format(exp_name=exp_name)
output_path = exp_cfg["response_path"].format(exp_name=exp_name)
mode = gen_cfg.get("mode", "vllm")

# Load data
queries = load_queries(data_cfg["queries_path"])
corpus = load_corpus(data_cfg["corpus_path"])

# Retrieval or with reranking
retriever = RunFileRetriever(
    run_path=retriever_cfg["run_path"],
    topk=retriever_cfg["topk"],
)
results = retriever.retrieve(queries, corpus)

# Context selection
max_model_len = gen_cfg.get("max_model_len", 10240)
truncate_hits(results, max_model_len=max_model_len, n_docs=sel_cfg["topk"])
select_top_k(results, sel_cfg["topk"])

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
responses = generator.generate(results, prompt_builder=prompt_builder)

# Write responses
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
with open(output_path, "w") as f:
    for qid, text in responses.items():
        f.write(json.dumps({"qid": qid, "response": text}) + "\n")
print(f"Wrote {len(responses)} responses to {output_path}")

# Write TREC context run
os.makedirs(os.path.dirname(context_path) or ".", exist_ok=True)
with open(context_path, "w") as f:
    for qid, result in results.items():
        for hit in result.hits:
            f.write(f"{qid} Q0 {hit.docid} {hit.rank} {hit.score} {exp_name}\n")
print(f"Wrote TREC run to {context_path}")
