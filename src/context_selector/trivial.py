from src.data import Result


def select_top_k(results: dict[str, Result], k: int) -> dict[str, Result]:
    for result in results.values():
        result.hits = result.hits[:k]
    return results
