import argparse
import json
import os
import glob
import yaml
from src.data import load_corpus, load_queries


def load_config(path: str, overrides: list[str]) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for override in overrides:
        key, _, value = override.partition("=")
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = yaml.safe_load(value)
    return cfg


def build_selector(cfg: dict, nugget_qrel_path: str = None):
    sel_cfg = cfg["selector"]
    sel_type = sel_cfg["type"]

    if sel_type == "top_k":
        from src.context_selector.top_k import TopKSelector
        return TopKSelector(k=sel_cfg["k"])

    # Nugget-based selectors: qrel path comes from selector config or caller (optimal mode)
    qrel_path = sel_cfg.get("nugget_qrel_path") or nugget_qrel_path
    if not qrel_path:
        raise ValueError(f"selector.type={sel_type} requires a nugget_qrel_path")

    from src.context_selector.greedy_nugget import (
        GreedyBudgetSelector,
        GreedyCompleteSelector,
        OracleAllSelector,
    )
    if sel_type == "greedy_budget":
        return GreedyBudgetSelector(nugget_qrel_path=qrel_path, max_docs=sel_cfg["k"])
    elif sel_type == "greedy_complete":
        return GreedyCompleteSelector(nugget_qrel_path=qrel_path)
    elif sel_type == "oracle_all":
        return OracleAllSelector(nugget_qrel_path=qrel_path)

    raise ValueError(f"Unknown selector type: {sel_type}")


def build_generator(cfg: dict):
    from src.generator.vllm_gen import VLLMGenerator
    return VLLMGenerator(**cfg["generator"])


def build_retriever(entry: dict):
    from src.retriever.run_file import RunFileRetriever
    return RunFileRetriever(
        run_path=entry["run_path"],
        topk=entry.get("topk", 100),
        name=entry.get("name", "run"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. pipeline=direct")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    mode = cfg["pipeline"]

    queries = load_queries(cfg["data"]["queries_path"])
    generator = build_generator(cfg)
    selector = build_selector(cfg)

    if mode == "direct":
        from src.pipeline.direct import DirectPipeline
        pipeline = DirectPipeline(generator=generator)
        responses = pipeline.run(queries)

    elif mode == "sequential":
        corpus = load_corpus(cfg["data"]["corpus_path"])
        retriever = build_retriever(cfg["retriever"][0])
        from src.pipeline.sequential import SequentialPipeline
        pipeline = SequentialPipeline(retriever=retriever, selector=selector, generator=generator)
        responses = pipeline.run(queries, corpus)

    elif mode == "parallel":
        corpus = load_corpus(cfg["data"]["corpus_path"])
        retrievers = [build_retriever(e) for e in cfg["retriever"]]
        from src.pipeline.parallel import ParallelPipeline
        pipeline = ParallelPipeline(
            retrievers=retrievers,
            selector=selector,
            generator=generator,
            rrf_k=cfg.get("rrf_k", 60),
        )
        responses = pipeline.run(queries, corpus)

    elif mode == "optimal":
        corpus = load_corpus(cfg["data"]["corpus_path"])
        nugget_qrel_pattern = cfg["data"].get("nugget_qrel_pattern", "data/nuggets/{qid}.qrel")
        nugget_qrel_paths = {
            qid: nugget_qrel_pattern.format(qid=qid) for qid in queries
        }
        selector_factory = lambda qrel_path: build_selector(cfg, nugget_qrel_path=qrel_path)
        from src.pipeline.optimal import OptimalPipeline
        pipeline = OptimalPipeline(selector_factory=selector_factory, generator=generator)
        responses = pipeline.run(queries, nugget_qrel_paths, corpus)

    else:
        raise ValueError(f"Unknown pipeline mode: {mode}")

    out_path = cfg["output"]["path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for qid, text in responses.items():
            f.write(json.dumps({"qid": qid, "response": text}) + "\n")

    print(f"Wrote {len(responses)} responses to {out_path}")


if __name__ == "__main__":
    main()
