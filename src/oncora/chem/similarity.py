"""Near-seen detection: max Tanimoto similarity to the training compounds (spec 03, Req 4; FR-4).

A prediction for a compound nearly identical to a training compound mostly reflects memory,
not generalization, so it is flagged.

Technique: Tanimoto similarity on bit fingerprints = |A ∧ B| / |A ∨ B| = shared on-bits divided
by the union of on-bits. It is computed over the bit-packed fingerprints as uint64 words with
NumPy's popcount: for 1M training compounds that is 256 MB, and one query takes well under a
second without building any RDKit objects.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oncora.config import NEAR_SEEN_TANIMOTO


def _as_words(packed: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    pad = (-packed.shape[-1]) % 8
    if pad:
        packed = np.pad(packed, [(0, 0)] * (packed.ndim - 1) + [(0, pad)])
    return packed.view(np.uint64)


def pack_bits(bits: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(bits, dtype=np.uint8), axis=-1)


def tanimoto(query_packed: np.ndarray, ref_packed: np.ndarray) -> np.ndarray:
    """Similarity of one packed query (n_bytes,) against packed references (n, n_bytes)."""
    q, r = _as_words(query_packed[None, :])[0], _as_words(ref_packed)
    inter = np.bitwise_count(r & q).sum(axis=1)
    union = np.bitwise_count(r | q).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, inter / union, 0.0)


@dataclass
class NearestNeighbor:
    max_tanimoto: float
    inchi_key: str | None
    near_seen: bool


class TrainingIndex:
    def __init__(self, keys: list[str], packed: np.ndarray, threshold: float = NEAR_SEEN_TANIMOTO):
        if len(keys) != len(packed):
            raise ValueError("keys and fingerprints differ in length")
        self.keys = np.asarray(keys)
        self.packed = np.ascontiguousarray(packed, dtype=np.uint8)
        self.threshold = threshold

    @classmethod
    def from_bytes(cls, keys: list[str], fps: list[bytes], **kw) -> "TrainingIndex":
        return cls(keys, np.frombuffer(b"".join(fps), dtype=np.uint8).reshape(len(fps), -1), **kw)

    def nearest(self, query_packed: np.ndarray) -> NearestNeighbor:
        if len(self.keys) == 0:
            return NearestNeighbor(0.0, None, False)
        sims = tanimoto(query_packed, self.packed)
        i = int(np.argmax(sims))
        best = float(sims[i])
        return NearestNeighbor(best, str(self.keys[i]), best >= self.threshold)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, keys=self.keys, packed=self.packed, threshold=self.threshold)

    @classmethod
    def load(cls, path: Path) -> "TrainingIndex":
        d = np.load(path, allow_pickle=False)
        return cls(d["keys"].tolist(), d["packed"], float(d["threshold"]))
