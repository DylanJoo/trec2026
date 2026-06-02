"""
Quick test script for interactive development. Modify parameters at the top and run:
  python test.py
"""

# ============================================================================
# CONFIGURATION - MODIFY HERE
# ============================================================================

TRACK = "rag"              # rag | biogen | ragtime
PIPELINE_MODE = "sequential"  # direct | sequential | parallel | optimal

# Selector
SELECTOR_TYPE = "top_k"   # top_k | greedy_budget | greedy_complete | oracle_all
SELECTOR_K = 3

# Generator
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
GEN_MAX_TOKENS = 256
GEN_NUM_GPUS = 1

# For parallel mode: which run files to fuse (keys into MOCK_RUNS below)
PARALLEL_RUNS = ["bm25", "cover"]
RRF_K = 60

# ============================================================================
# MOCK DATA PER TRACK
# ============================================================================

MOCK_DATA = {
    "rag": {
        "queries": {
            "2027497": {
                "query": "What causes aurora borealis?",
                "meta": {},
            },
            "2027498": {
                "query": "How does mRNA vaccine technology work?",
                "meta": {},
            },
        },
        "corpus": {
            "seg_001": {"text": "The aurora borealis is caused by charged particles from the sun interacting with Earth's magnetic field.", "title": "Aurora Overview", "url": "", "meta": {}},
            "seg_002": {"text": "Solar wind carries electrons and protons that collide with atmospheric gases, producing colorful light emissions.", "title": "Solar Wind Effects", "url": "", "meta": {}},
            "seg_003": {"text": "mRNA vaccines work by introducing messenger RNA into cells, instructing them to produce a protein that triggers an immune response.", "title": "mRNA Vaccines", "url": "", "meta": {}},
            "seg_004": {"text": "The immune system recognizes the spike protein produced from vaccine mRNA and builds memory cells for future defense.", "title": "Immune Response", "url": "", "meta": {}},
        },
        "runs": {
            "bm25": {"2027497": [("seg_001", 9.1), ("seg_002", 7.5)], "2027498": [("seg_003", 8.8), ("seg_004", 7.2)]},
            "cover": {"2027497": [("seg_002", 8.9), ("seg_001", 7.0)], "2027498": [("seg_004", 8.5), ("seg_003", 7.9)]},
        },
        "nuggets": {
            "2027497": {"seg_001": {"nug_solar_particles"}, "seg_002": {"nug_magnetic_field", "nug_light_emission"}},
            "2027498": {"seg_003": {"nug_mrna_mechanism"}, "seg_004": {"nug_immune_memory", "nug_spike_protein"}},
        },
    },
    "biogen": {
        "queries": {
            "BG001": {
                "query": "What are the side effects of metformin in type 2 diabetes patients?",
                "meta": {},
            },
            "BG002": {
                "query": "How effective is cognitive behavioral therapy for depression?",
                "meta": {},
            },
        },
        "corpus": {
            "36001234": {"text": "Metformin is generally well tolerated. Common side effects include gastrointestinal symptoms such as nausea, diarrhea, and abdominal discomfort.", "title": "Metformin tolerability", "url": "https://pubmed.ncbi.nlm.nih.gov/36001234/", "meta": {"pmid": "36001234"}},
            "35997654": {"text": "Lactic acidosis is a rare but serious complication associated with metformin use, particularly in patients with renal impairment.", "title": "Metformin and lactic acidosis", "url": "https://pubmed.ncbi.nlm.nih.gov/35997654/", "meta": {"pmid": "35997654"}},
            "35880012": {"text": "Cognitive behavioral therapy (CBT) has demonstrated efficacy comparable to antidepressants in treating mild-to-moderate depression.", "title": "CBT for depression", "url": "https://pubmed.ncbi.nlm.nih.gov/35880012/", "meta": {"pmid": "35880012"}},
            "35760543": {"text": "Meta-analyses show that combined CBT and pharmacotherapy produces better long-term outcomes than either treatment alone.", "title": "Combined depression treatment", "url": "https://pubmed.ncbi.nlm.nih.gov/35760543/", "meta": {"pmid": "35760543"}},
        },
        "runs": {
            "bm25": {"BG001": [("36001234", 9.5), ("35997654", 8.1)], "BG002": [("35880012", 9.3), ("35760543", 8.7)]},
            "cover": {"BG001": [("35997654", 8.8), ("36001234", 7.5)], "BG002": [("35760543", 8.9), ("35880012", 8.0)]},
        },
        "nuggets": {
            "BG001": {"36001234": {"nug_gi_effects"}, "35997654": {"nug_lactic_acidosis", "nug_renal_risk"}},
            "BG002": {"35880012": {"nug_cbt_efficacy"}, "35760543": {"nug_combined_treatment"}},
        },
    },
    "ragtime": {
        "queries": {
            "RT001": {
                "query": "Impact of artificial intelligence on global labor markets",
                "meta": {"background": "I am a policy researcher studying employment trends.", "report_length": 2000},
            },
            "RT002": {
                "query": "Climate adaptation strategies in coastal cities",
                "meta": {"background": "I am a city planner assessing flood risk mitigation.", "report_length": 2000},
            },
        },
        "corpus": {
            "news_en_001": {"text": "AI automation is projected to displace up to 85 million jobs by 2025, while creating 97 million new roles in emerging sectors.", "title": "AI and Jobs Report", "url": "", "meta": {"lang": "en"}},
            "news_zh_001": {"text": "人工智能技术的快速发展正在改变全球劳动力市场格局，尤其在制造业和服务业产生深远影响。", "title": "AI与就业市场", "url": "", "meta": {"lang": "zh"}},
            "news_en_002": {"text": "Coastal cities are implementing green infrastructure such as mangrove restoration and sea walls to combat rising sea levels.", "title": "Coastal Adaptation", "url": "", "meta": {"lang": "en"}},
            "news_ar_001": {"text": "تواجه المدن الساحلية تحديات جسيمة بسبب ارتفاع منسوب البحر، مما يستدعي استراتيجيات تكيف فعّالة.", "title": "المدن الساحلية والتكيف", "url": "", "meta": {"lang": "ar"}},
        },
        "runs": {
            "bm25": {"RT001": [("news_en_001", 9.2), ("news_zh_001", 7.8)], "RT002": [("news_en_002", 9.0), ("news_ar_001", 7.5)]},
            "cover": {"RT001": [("news_zh_001", 8.5), ("news_en_001", 8.0)], "RT002": [("news_ar_001", 8.2), ("news_en_002", 8.8)]},
        },
        "nuggets": {
            "RT001": {"news_en_001": {"nug_job_displacement", "nug_new_roles"}, "news_zh_001": {"nug_manufacturing_impact"}},
            "RT002": {"news_en_002": {"nug_green_infrastructure", "nug_sea_level"}, "news_ar_001": {"nug_coastal_challenge"}},
        },
    },
}

# ============================================================================
# SETUP — write mock data to temp files
# ============================================================================

import os
import json
import tempfile
from pathlib import Path


def setup_mock_data(track: str) -> Path:
    data = MOCK_DATA[track]
    tmpdir = Path(tempfile.gettempdir()) / f"trec2026_test_{track}"
    tmpdir.mkdir(exist_ok=True)

    # Queries JSONL (unified format)
    queries_file = tmpdir / "queries.jsonl"
    with open(queries_file, "w") as f:
        for qid, q in data["queries"].items():
            f.write(json.dumps({"qid": qid, "query": q["query"], "meta": q["meta"]}) + "\n")

    # Corpus JSONL (unified format)
    corpus_file = tmpdir / "corpus.jsonl"
    with open(corpus_file, "w") as f:
        for docid, doc in data["corpus"].items():
            f.write(json.dumps({"docid": docid, **doc}) + "\n")

    # Run files
    runs_dir = tmpdir / "runs"
    runs_dir.mkdir(exist_ok=True)
    for run_name, run_dict in data["runs"].items():
        with open(runs_dir / f"{run_name}.trec", "w") as f:
            for qid, hits in run_dict.items():
                for rank, (docid, score) in enumerate(hits, start=1):
                    f.write(f"{qid} Q0 {docid} {rank} {score:.1f} {run_name}\n")

    # Nugget qrels (per-query files)
    nuggets_dir = tmpdir / "nuggets"
    nuggets_dir.mkdir(exist_ok=True)
    for qid, doc_nugsets in data["nuggets"].items():
        with open(nuggets_dir / f"{qid}.qrel", "w") as f:
            for docid, nugsets in doc_nugsets.items():
                for nugid in nugsets:
                    f.write(f"{docid} {nugid} 1\n")

    return tmpdir


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    tmpdir = setup_mock_data(TRACK)
    print(f"Track: {TRACK} | Mode: {PIPELINE_MODE} | Selector: {SELECTOR_TYPE}")
    print(f"Mock data: {tmpdir}\n")

    from src.data import load_queries, load_corpus
    from src.generator.vllm_gen import VLLMGenerator

    queries = load_queries(str(tmpdir / "queries.jsonl"))
    corpus = load_corpus(str(tmpdir / "corpus.jsonl"))

    generator = VLLMGenerator(
        model_name_or_path=MODEL_NAME,
        track=TRACK,
        max_tokens=GEN_MAX_TOKENS,
        num_gpus=GEN_NUM_GPUS,
    )

    def build_selector(qrel_path=None):
        if SELECTOR_TYPE == "top_k":
            from src.context_selector.top_k import TopKSelector
            return TopKSelector(k=SELECTOR_K)
        from src.context_selector.greedy_nugget import (
            GreedyBudgetSelector, GreedyCompleteSelector, OracleAllSelector
        )
        if SELECTOR_TYPE == "greedy_budget":
            return GreedyBudgetSelector(nugget_qrel_path=qrel_path, max_docs=SELECTOR_K)
        elif SELECTOR_TYPE == "greedy_complete":
            return GreedyCompleteSelector(nugget_qrel_path=qrel_path)
        elif SELECTOR_TYPE == "oracle_all":
            return OracleAllSelector(nugget_qrel_path=qrel_path)
        raise ValueError(SELECTOR_TYPE)

    # ------------------------------------------------------------------
    if PIPELINE_MODE == "direct":
        from src.pipeline.direct import DirectPipeline
        responses = DirectPipeline(generator=generator).run(queries)

    elif PIPELINE_MODE == "sequential":
        from src.pipeline.sequential import SequentialPipeline
        from src.retriever.run_file import RunFileRetriever
        retriever = RunFileRetriever(run_path=str(tmpdir / "runs/bm25.trec"))
        responses = SequentialPipeline(
            retriever=retriever, selector=build_selector(), generator=generator
        ).run(queries, corpus)

    elif PIPELINE_MODE == "parallel":
        from src.pipeline.parallel import ParallelPipeline
        from src.retriever.run_file import RunFileRetriever
        retrievers = [
            RunFileRetriever(run_path=str(tmpdir / f"runs/{name}.trec"), name=name)
            for name in PARALLEL_RUNS
        ]
        responses = ParallelPipeline(
            retrievers=retrievers, selector=build_selector(), generator=generator, rrf_k=RRF_K
        ).run(queries, corpus)

    elif PIPELINE_MODE == "optimal":
        from src.pipeline.optimal import OptimalPipeline
        nugget_paths = {qid: str(tmpdir / f"nuggets/{qid}.qrel") for qid in queries}
        responses = OptimalPipeline(
            selector_factory=lambda p: build_selector(qrel_path=p),
            generator=generator,
        ).run(queries, nugget_paths, corpus)

    else:
        raise ValueError(PIPELINE_MODE)

    # ------------------------------------------------------------------
    print("=== Responses ===")
    for qid, resp in responses.items():
        query = queries[qid]["query"]
        print(f"\n[{qid}] {query}")
        print(f"  {resp[:300]}{'...' if len(resp) > 300 else ''}")

    print(f"\nDone — {len(responses)} responses")
