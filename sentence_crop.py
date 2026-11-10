"""Fixed, deterministic sentence-window crop used by the round-3 experiment."""

from __future__ import annotations

import hashlib
import random
import re


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+")
MIN_ORIGINAL_WORDS = 120
MIN_CROP_WORDS = 40
MIN_RATIO = 0.65
MAX_RATIO = 0.85
CROP_PROBABILITY = 0.5


def candidate_spans(text: str) -> list[tuple[int, int]]:
    """Return eligible contiguous sentence-window offsets in the original text."""
    original_words = len(text.split())
    if original_words < MIN_ORIGINAL_WORDS:
        return []
    bounds = [0, *(match.end() for match in SENTENCE_BOUNDARY.finditer(text)), len(text)]
    n_sentences = len(bounds) - 1
    spans: list[tuple[int, int]] = []
    for left in range(n_sentences):
        for right in range(left + 1, n_sentences + 1):
            if left == 0 and right == n_sentences:
                continue
            span = text[bounds[left]:bounds[right]]
            words = len(span.split())
            if words >= MIN_CROP_WORDS and MIN_RATIO <= words / original_words <= MAX_RATIO:
                spans.append((bounds[left], bounds[right]))
    return spans


def crop_for_epoch(text: str, spans: list[tuple[int, int]], *, seed: int,
                   outer_fold: int, epoch: int, row_id: str) -> tuple[str, bool]:
    """Choose from one ID's fixed candidates without touching global RNG state."""
    if not spans:
        return text, False
    key = f"{seed}:{outer_fold}:{epoch}:{row_id}:crop".encode("utf-8")
    random_seed = int(hashlib.sha256(key).hexdigest()[:16], 16)
    rng = random.Random(random_seed)
    if rng.random() >= CROP_PROBABILITY:
        return text, False
    left, right = spans[rng.randrange(len(spans))]
    cropped = text[left:right]
    if cropped == text or not (MIN_CROP_WORDS <= len(cropped.split())):
        raise AssertionError("Invalid crop produced")
    return cropped, True
