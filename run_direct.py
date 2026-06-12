"""Direct generation: no retrieval, pass empty context.

Example usage:
  python run_direct.py config/ragtime.yaml
"""
import json
import os
import sys

import torch
import yaml

from src.generator import VLLMGenerator, EndpointGenerator
from src.generator.prompt import get_prompt_builder
from src.data import Result, load_queries

# Load configurations
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg["data"]
gen_cfg = cfg["generator"]
exp_cfg = cfg["exp"]

exp_name = exp_cfg["exp_name"]
output_path = exp_cfg["response_path"].format(exp_name=exp_name)
mode = gen_cfg.get("mode", "vllm")

# Input preparation
queries = load_queries(data_cfg["queries_path"])

# Retrieval-augmented Context
result_list = [Result(qid=qid, query=q["query"], meta=q.get("meta", {})) for qid, q in queries.items()]
contexts = {r.qid: [] for r in result_list}

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

# Write
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
with open(output_path, "w") as f:
    for qid, text in responses.items():
        f.write(json.dumps({"qid": qid, "response": text}) + "\n")
print(f"Wrote {len(responses)} responses to {output_path}")
