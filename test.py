"""
Quick test script for development. Modify parameters at the top and run:
  python test.py
"""

# ============================================================================
# CONFIGURATION - MODIFY HERE
# ============================================================================

PIPELINE_MODE = "sequential"  # direct | sequential | parallel | optimal

# Data paths (use mock data if files don't exist)
QUERIES_PATH = "data/queries.tsv"
CORPUS_PATH = "data/corpus.jsonl"

# Retriever (for sequential/parallel modes)
RUN_FILES = {
    "bm25": "runs/bm25.trec",
    "cover": "runs/cover.trec",
}

# Selector
SELECTOR_TYPE = "top_k"  # top_k | greedy_budget | greedy_complete | oracle_all
SELECTOR_K = 5  # for top_k and greedy_budget
NUGGET_QREL_PATH = None  # filled per-query in optimal mode, format: data/nuggets/{qid}.qrel

# Generator
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
GEN_TEMPERATURE = 0.0
GEN_MAX_TOKENS = 512
GEN_NUM_GPUS = 1

# RRF (for parallel mode)
RRF_K = 60

# ============================================================================
# MOCK DATA (if files don't exist)
# ============================================================================

MOCK_QUERIES = {
    "q1": "What is machine learning?",
    "q2": "Explain neural networks",
}

MOCK_CORPUS = {
    "d1": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without explicit programming.",
    "d2": "Neural networks are computing systems inspired by biological neural networks that constitute animal brains.",
    "d3": "Deep learning uses multiple layers of artificial neurons to process complex patterns in data.",
    "d4": "Supervised learning requires labeled training data with known outcomes.",
    "d5": "Unsupervised learning finds patterns in unlabeled data without predefined outcomes.",
}

MOCK_RUN_BM25 = {
    "q1": [("d1", 8.5), ("d4", 7.2), ("d5", 6.1)],
    "q2": [("d2", 9.1), ("d3", 8.3), ("d1", 5.2)],
}

MOCK_RUN_COVER = {
    "q1": [("d2", 8.2), ("d1", 7.9), ("d3", 6.5)],
    "q2": [("d3", 9.0), ("d2", 8.5), ("d4", 5.0)],
}

MOCK_NUGGET_QRELS = {
    "q1": {
        "d1": {"nug_ml_def", "nug_ai"},
        "d2": {"nug_nn_def"},
        "d3": {"nug_dl"},
        "d4": {"nug_supervised"},
    },
    "q2": {
        "d2": {"nug_nn_def", "nug_nn_bio"},
        "d3": {"nug_dl", "nug_layers"},
        "d4": {"nug_supervised"},
    },
}

# ============================================================================
# SETUP
# ============================================================================

import os
import sys
import json
import tempfile
from pathlib import Path


def setup_mock_data():
    """Create temporary mock data files if originals don't exist."""
    tmpdir = Path(tempfile.gettempdir()) / "trec2026_test"
    tmpdir.mkdir(exist_ok=True)

    # Queries
    queries_file = tmpdir / "queries.tsv"
    if not queries_file.exists():
        with open(queries_file, "w") as f:
            for qid, text in MOCK_QUERIES.items():
                f.write(f"{qid}\t{text}\n")

    # Corpus
    corpus_file = tmpdir / "corpus.jsonl"
    if not corpus_file.exists():
        with open(corpus_file, "w") as f:
            for docid, text in MOCK_CORPUS.items():
                f.write(json.dumps({"docid": docid, "contents": text}) + "\n")

    # Run files
    runs_dir = tmpdir / "runs"
    runs_dir.mkdir(exist_ok=True)

    def write_run(run_dict, name):
        run_file = runs_dir / f"{name}.trec"
        if not run_file.exists():
            with open(run_file, "w") as f:
                for qid, hits in run_dict.items():
                    for rank, (docid, score) in enumerate(hits, start=1):
                        f.write(f"{qid} Q0 {docid} {rank} {score:.1f} {name}\n")
        return str(run_file)

    write_run(MOCK_RUN_BM25, "bm25")
    write_run(MOCK_RUN_COVER, "cover")

    # Nugget qrels (per-query files)
    nuggets_dir = tmpdir / "nuggets"
    nuggets_dir.mkdir(exist_ok=True)

    for qid, doc_nuggets in MOCK_NUGGET_QRELS.items():
        qrel_file = nuggets_dir / f"{qid}.qrel"
        if not qrel_file.exists():
            with open(qrel_file, "w") as f:
                for docid, nugsets in doc_nuggets.items():
                    for nugid in nugsets:
                        f.write(f"{docid} {nugid} 1\n")

    return tmpdir


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    tmpdir = setup_mock_data()
    print(f"Using mock data in: {tmpdir}\n")

    # Override paths
    queries_path = str(tmpdir / "queries.tsv")
    corpus_path = str(tmpdir / "corpus.jsonl")
    nugget_qrel_pattern = str(tmpdir / "nuggets" / "{qid}.qrel")

    # Load data
    from src.data import load_queries, load_corpus, load_run

    queries = load_queries(queries_path)
    corpus = load_corpus(corpus_path)

    print(f"Loaded {len(queries)} queries, {len(corpus)} docs\n")

    # Build generator
    from src.generator.vllm_gen import VLLMGenerator

    generator = VLLMGenerator(
        model_name_or_path=MODEL_NAME,
        temperature=GEN_TEMPERATURE,
        max_tokens=GEN_MAX_TOKENS,
        num_gpus=GEN_NUM_GPUS,
    )

    # ========================================================================
    # DIRECT
    # ========================================================================
    if PIPELINE_MODE == "direct":
        print("=== DIRECT MODE (no retrieval) ===\n")
        from src.pipeline.direct import DirectPipeline

        pipeline = DirectPipeline(generator=generator)
        responses = pipeline.run(queries)

        for qid, resp in responses.items():
            print(f"Q{qid}:")
            print(f"  {resp[:200]}...\n")

    # ========================================================================
    # SEQUENTIAL
    # ========================================================================
    elif PIPELINE_MODE == "sequential":
        print(f"=== SEQUENTIAL MODE ({SELECTOR_TYPE} selector) ===\n")

        from src.pipeline.sequential import SequentialPipeline
        from src.retriever.run_file import RunFileRetriever
        from src.context_selector.top_k import TopKSelector

        run_path = str(tmpdir / f"runs/{list(RUN_FILES.keys())[0]}.trec")
        retriever = RunFileRetriever(run_path=run_path, topk=100)
        selector = TopKSelector(k=SELECTOR_K)

        pipeline = SequentialPipeline(
            retriever=retriever, selector=selector, generator=generator
        )
        responses = pipeline.run(queries, corpus)

        for qid, resp in responses.items():
            print(f"Q{qid}:")
            print(f"  {resp[:200]}...\n")

    # ========================================================================
    # PARALLEL
    # ========================================================================
    elif PIPELINE_MODE == "parallel":
        print(f"=== PARALLEL MODE (RRF + {SELECTOR_TYPE} selector) ===\n")

        from src.pipeline.parallel import ParallelPipeline
        from src.retriever.run_file import RunFileRetriever
        from src.context_selector.top_k import TopKSelector

        retrievers = [
            RunFileRetriever(
                run_path=str(tmpdir / f"runs/{name}.trec"),
                topk=100,
                name=name,
            )
            for name in RUN_FILES.keys()
        ]
        selector = TopKSelector(k=SELECTOR_K)

        pipeline = ParallelPipeline(
            retrievers=retrievers,
            selector=selector,
            generator=generator,
            rrf_k=RRF_K,
        )
        responses = pipeline.run(queries, corpus)

        for qid, resp in responses.items():
            print(f"Q{qid}:")
            print(f"  {resp[:200]}...\n")

    # ========================================================================
    # OPTIMAL
    # ========================================================================
    elif PIPELINE_MODE == "optimal":
        print(f"=== OPTIMAL MODE ({SELECTOR_TYPE} selector) ===\n")

        from src.pipeline.optimal import OptimalPipeline

        def build_selector(qrel_path):
            if SELECTOR_TYPE == "greedy_budget":
                from src.context_selector.greedy_nugget import GreedyBudgetSelector

                return GreedyBudgetSelector(
                    nugget_qrel_path=qrel_path, max_docs=SELECTOR_K
                )
            elif SELECTOR_TYPE == "greedy_complete":
                from src.context_selector.greedy_nugget import GreedyCompleteSelector

                return GreedyCompleteSelector(nugget_qrel_path=qrel_path)
            elif SELECTOR_TYPE == "oracle_all":
                from src.context_selector.greedy_nugget import OracleAllSelector

                return OracleAllSelector(nugget_qrel_path=qrel_path)
            else:
                raise ValueError(f"Unknown optimal selector: {SELECTOR_TYPE}")

        nugget_qrel_paths = {qid: nugget_qrel_pattern.format(qid=qid) for qid in queries}
        pipeline = OptimalPipeline(selector_factory=build_selector, generator=generator)
        responses = pipeline.run(queries, nugget_qrel_paths, corpus)

        for qid, resp in responses.items():
            print(f"Q{qid}:")
            print(f"  {resp[:200]}...\n")

    else:
        raise ValueError(f"Unknown pipeline mode: {PIPELINE_MODE}")

    print("✓ Done")
