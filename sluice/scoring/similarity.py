"""Lightweight text similarity: word-frequency cosine similarity, no new dependency.

Used to pick the most relevant past headlines for the novelty question, and to
shrink a news item's own summary when a Jev request is otherwise too large.
"""

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _tokens(text: str) -> Counter[str]:
    return Counter(_TOKEN_RE.findall(text.lower()))


def similarity(a: str, b: str) -> float:
    """Cosine similarity of word-count vectors. 0.0 when either side has no tokens."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    dot = sum(count * tb[word] for word, count in ta.items() if word in tb)
    norm = math.sqrt(sum(c * c for c in ta.values())) * math.sqrt(sum(c * c for c in tb.values()))
    return dot / norm if norm else 0.0


def most_similar(query: str, candidates: list[str], k: int) -> list[str]:
    """The `k` candidates most similar to `query`, most similar first.

    Ties keep the candidates' input order (stable sort on a negated score).
    """
    ranked = sorted(range(len(candidates)), key=lambda i: -similarity(query, candidates[i]))
    return [candidates[i] for i in ranked[:k]]


def split_sentences(text: str) -> list[str]:
    """Naive split on sentence-ending punctuation; good enough for short summaries."""
    return [s for s in _SENTENCE_RE.split(text.strip()) if s]
