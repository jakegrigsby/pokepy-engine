"""Pokemon text tokenizer.

Loads metamon's `DefaultObservationSpace-v1.json` (2541 entries), splits text
on whitespace, and maps words to token IDs. Unknown words become -1
(metamon's UNKNOWN_TOKEN convention).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

UNKNOWN_TOKEN = -1

class PokemonTokenizer:
    """Whitespace-split + dict-lookup tokenizer."""

    __slots__ = ("vocab",)

    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.vocab)

    def tokenize_words(self, words: List[str]) -> np.ndarray:
        ids = [self.vocab.get(w, UNKNOWN_TOKEN) for w in words]
        return np.asarray(ids, dtype=np.int32)

    def tokenize(self, text: str) -> np.ndarray:
        return self.tokenize_words(text.split())

_DEFAULT_VOCAB_PATH = Path(__file__).parent / "data" / "DefaultObservationSpace-v1.json"
_default_tokenizer: PokemonTokenizer | None = None

def load_default_tokenizer() -> PokemonTokenizer:
    global _default_tokenizer
    if _default_tokenizer is not None:
        return _default_tokenizer
    with open(_DEFAULT_VOCAB_PATH) as f:
        vocab = json.load(f)
    _default_tokenizer = PokemonTokenizer(vocab)
    return _default_tokenizer
