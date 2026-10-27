"""Reciprocal Rank Fusion, implemented by us (custom algorithm #2).

Semantic and keyword search produce scores on different scales (cosine similarity vs BM25), so we
combine their RANKS, not their scores:

    RRF(d) = sum over result lists L containing d of   w_L / (k + rank_L(d))

- rank_L(d) is d's 1-based position in list L
- k = 60 (the value from the original RRF paper) damps the difference between top ranks, so one
  list cannot dominate just by putting d first
- w_L is an optional weight per list (default 1.0 each)

A chunk found by both searches collects two terms, so agreement between the methods is rewarded.
Duplicates inside one list count only at their best rank. Ties are broken by the best single rank,
then by id, so the output is fully deterministic.

Hand example (k = 60): semantic = [A, B, C], keyword = [B, D]
    A = 1/61          = 0.016393
    B = 1/62 + 1/61   = 0.032522   (found by both -> first)
    C = 1/63          = 0.015873
    D = 1/62          = 0.016129
    result: B, A, D, C
"""


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking is required")
    if k <= 0:
        raise ValueError("k must be positive")

    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking, weight in zip(rankings, weights):
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], kv[0]))
