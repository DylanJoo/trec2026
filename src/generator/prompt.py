from dataclasses import dataclass, field
from typing import Callable
from src.data import Hit


# ---------------------------------------------------------------------------
# TREC RAG 2024/2025
# Task: answer a narrative question with citations to MS MARCO v2.1 segments
# Output: up to 400 words, sentence-level with inline [N] citations
# ---------------------------------------------------------------------------

RAG_SYSTEM = (
    "You are a helpful assistant. Answer the question using only the provided passages. "
    "Write up to 400 words. Break your answer into sentences. "
    "After each sentence, cite the passage number(s) that support it, e.g. [1][3]."
)

def build_rag_prompt(query: str, hits: list[Hit], meta: dict) -> list[dict]:
    passages = "\n\n".join(
        f"[{i+1}] {h.content}" for i, h in enumerate(hits)
    )
    return [
        {"role": "system", "content": RAG_SYSTEM},
        {"role": "user", "content": f"Passages:\n{passages}\n\nQuestion: {query}\n\nAnswer:"},
    ]


# ---------------------------------------------------------------------------
# TREC BioGen 2024/2025
# Task: answer a biomedical question with citations to PubMed abstracts (PMIDs)
# Output: sentence-level answer, each sentence citing supporting PMIDs
# ---------------------------------------------------------------------------

BIOGEN_SYSTEM = (
    "You are a biomedical expert. Answer the medical question based only on the provided "
    "PubMed abstracts. Write a clear answer. After each sentence, cite the PMID(s) that "
    "support it, e.g. [PMID: 12345678]."
)

def build_biogen_prompt(query: str, hits: list[Hit], meta: dict) -> list[dict]:
    abstracts = "\n\n".join(
        f"[PMID: {h.docid}] {h.content}" for h in hits
    )
    return [
        {"role": "system", "content": BIOGEN_SYSTEM},
        {"role": "user", "content": f"Abstracts:\n{abstracts}\n\nQuestion: {query}\n\nAnswer:"},
    ]


# ---------------------------------------------------------------------------
# TREC RAGTIME 2025
# Task: generate a multilingual report from a problem statement + background
# Output: ~2000 or ~10000 char report with inline citations
# ---------------------------------------------------------------------------

RAGTIME_SYSTEM = (
    "You are a research analyst. Write a well-structured report that addresses the given "
    "information need. Use only the provided documents as sources. "
    "Cite each document as [N] after the sentence it supports."
)

def build_ragtime_prompt(query: str, hits: list[Hit], meta: dict) -> list[dict]:
    background = meta.get("background", "")
    report_length = meta.get("report_length", 2000)

    docs = "\n\n".join(
        f"[{i+1}] ({h.meta.get('lang', '?')}) {h.title + ': ' if h.title else ''}{h.content}"
        for i, h in enumerate(hits)
    )
    user_content = ""
    if background:
        user_content += f"Background: {background}\n\n"
    user_content += f"Documents:\n{docs}\n\n"
    user_content += f"Report request: {query}\n\n"
    user_content += f"Write a report of approximately {report_length} characters:\n"

    return [
        {"role": "system", "content": RAGTIME_SYSTEM},
        {"role": "user", "content": user_content},
    ]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PROMPT_BUILDERS: dict[str, Callable] = {
    "rag":     build_rag_prompt,
    "biogen":  build_biogen_prompt,
    "ragtime": build_ragtime_prompt,
}

def get_prompt_builder(track: str) -> Callable:
    if track not in PROMPT_BUILDERS:
        raise ValueError(f"Unknown track '{track}'. Choose from: {list(PROMPT_BUILDERS)}")
    return PROMPT_BUILDERS[track]
