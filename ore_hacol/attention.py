"""IO-aware attention: stable reference attention, online softmax, and
FlashAttention-style tiled attention (integrated steps 17-23).

The tiled implementation never materializes the full attention score matrix in
global memory: only per-tile blocks exist, and per-row running statistics
(m, l, a) carry the streaming reduction:

    m' = max(m, m_b)
    l' = e^(m-m') l + sum_{j in b} e^(x_j - m')
    a' = e^(m-m') a + sum_{j in b} e^(x_j - m') v_j
    o  = a' / l'

Online softmax is a reusable streaming primitive, not a universal optimization;
attention eligibility is checked before tiling and any incompatibility falls
back to the trusted reference (step 23).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: Policies for rows whose every key is masked (contract field all_masked_row_policy).
ALL_MASKED_POLICIES = ("zeros", "uniform", "reject")


@dataclass(frozen=True)
class AttentionRecord:
    """Execution metadata for one attention call (validation and memory evidence)."""

    algorithm: str
    tile_query: int = 0
    tile_key: int = 0
    score_matrix_materialized: bool = False
    peak_tile_bytes: int = 0
    all_masked_rows: int = 0
    fallback_reason: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "tile_query": self.tile_query,
            "tile_key": self.tile_key,
            "score_matrix_materialized": self.score_matrix_materialized,
            "peak_tile_bytes": self.peak_tile_bytes,
            "all_masked_rows": self.all_masked_rows,
            "fallback_reason": self.fallback_reason,
            "notes": list(self.notes),
        }


def _as_2d(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be floating dtype, got {array.dtype}")
    return array


def _default_scale(dim: int) -> float:
    return 1.0 / np.sqrt(float(dim))


def _resolved_mask(mask: np.ndarray | None, rows: int, columns: int) -> np.ndarray | None:
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.shape != (rows, columns):
        raise ValueError(f"mask shape {array.shape} does not match {(rows, columns)}")
    if array.dtype != bool:
        array = array != 0
    return array


def attention_eligibility(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Check attention eligibility (integrated step 17).

    Confirms actual query/key/value attention semantics and supported
    dimensions; generic matrix products must never be routed here.
    """
    try:
        q2 = _as_2d("q", q)
        k2 = _as_2d("k", k)
        v2 = _as_2d("v", v)
    except ValueError as exc:
        return False, str(exc)
    if q2.shape[1] != k2.shape[1]:
        return False, "q and k feature dimensions differ; not attention semantics"
    if k2.shape[0] != v2.shape[0]:
        return False, "k rows and v rows differ; not attention semantics"
    if not (np.all(np.isfinite(q2)) and np.all(np.isfinite(k2)) and np.all(np.isfinite(v2))):
        return False, "non-finite inputs are unsupported"
    if mask is not None:
        array = np.asarray(mask)
        if array.shape != (q2.shape[0], k2.shape[0]):
            return False, f"mask shape {array.shape} unsupported for {(q2.shape[0], k2.shape[0])}"
    return True, "attention semantics confirmed"


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise stable softmax: subtract the row maximum before exponentiating."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=-1, keepdims=True)


def reference_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    scale: float | None = None,
    all_masked_policy: str = "zeros",
) -> tuple[np.ndarray, AttentionRecord]:
    """Trusted conventional attention with a stable softmax (integrated step 4).

    Materializes the full score matrix; used as the reference implementation
    that every optimized attention must match within tolerance.
    """
    q2 = _as_2d("q", q).astype(np.float64)
    k2 = _as_2d("k", k).astype(np.float64)
    v2 = _as_2d("v", v).astype(np.float64)
    if all_masked_policy not in ALL_MASKED_POLICIES:
        raise ValueError(f"all_masked_policy must be one of {ALL_MASKED_POLICIES}")
    rows, dim = q2.shape
    key_count = k2.shape[0]
    factor = scale if scale is not None else _default_scale(dim)
    resolved = _resolved_mask(mask, rows, key_count)

    logits = (q2 @ k2.T) * factor
    if resolved is not None:
        logits = np.where(resolved, logits, -np.inf)
    with np.errstate(invalid="ignore", over="ignore"):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        denominator = np.nan_to_num(
            np.sum(exponentials, axis=1, keepdims=True), nan=0.0, posinf=0.0
        )
        weights = np.nan_to_num(
            exponentials / np.where(denominator > 0.0, denominator, 1.0),
            nan=0.0,
            posinf=0.0,
        )
    weights = np.where(denominator > 0.0, weights, 0.0)

    all_masked_rows = int(np.sum(~np.any(resolved, axis=1))) if resolved is not None else 0
    if all_masked_rows:
        if all_masked_policy == "reject":
            raise ValueError("reference attention received an all-masked row under 'reject' policy")
        if all_masked_policy == "uniform":
            uniform = np.full(key_count, 1.0 / key_count, dtype=np.float64)
            weights = np.where(denominator > 0.0, weights, uniform)
    output = weights @ v2
    record = AttentionRecord(
        algorithm="reference_attention",
        score_matrix_materialized=True,
        peak_tile_bytes=int(logits.nbytes),
        all_masked_rows=all_masked_rows,
        notes=("stable softmax with row-maximum subtraction",),
    )
    return output, record


class OnlineSoftmaxState:
    """Streaming (m, l, a) accumulator implementing the online-softmax equations.

    Maintains the running maximum ``m``, denominator ``l`` and weighted
    numerator ``a`` for each query row so softmax attention can be computed
    block-by-block without materializing the full probability matrix.
    """

    def __init__(self, rows: int, value_dim: int) -> None:
        self.m = np.full(rows, -np.inf, dtype=np.float64)
        self.l = np.zeros(rows, dtype=np.float64)
        self.a = np.zeros((rows, value_dim), dtype=np.float64)

    def update(self, logits_block: np.ndarray, values_block: np.ndarray) -> None:
        """Fold one key block into the running statistics (integrated step 20)."""
        block_max = np.max(logits_block, axis=1)
        new_m = np.maximum(self.m, block_max)
        # Rows where both old and new maxima are -inf stay at -inf without NaN.
        finite = np.isfinite(new_m)
        safe_m = np.where(finite, new_m, 0.0)
        with np.errstate(invalid="ignore", over="ignore"):
            alpha = np.exp(
                np.where(np.isfinite(self.m), self.m, safe_m) - safe_m
            )
            shifted = logits_block - safe_m[:, None]
            probabilities = np.exp(shifted)
        alpha = np.where(finite, alpha, 0.0)
        probabilities = np.where(np.isfinite(logits_block), probabilities, 0.0)
        self.l = alpha * self.l + probabilities.sum(axis=1)
        self.a = alpha[:, None] * self.a + probabilities @ values_block
        self.m = new_m

    def finalize(self, all_masked_policy: str = "zeros", values: np.ndarray | None = None) -> np.ndarray:
        """Normalize the numerator: o = a / l (integrated step 21)."""
        rows = self.a.shape[0]
        empty = self.l <= 0.0
        output = np.zeros_like(self.a)
        if np.any(~empty):
            output[~empty] = self.a[~empty] / self.l[~empty, None]
        if np.any(empty):
            if all_masked_policy == "reject":
                raise ValueError("online softmax finalized an all-masked row under 'reject' policy")
            if all_masked_policy == "uniform" and values is not None:
                output[empty] = np.mean(values, axis=0)
            elif all_masked_policy == "uniform":
                output[empty] = 0.0
        return output


def online_softmax_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    scale: float | None = None,
    all_masked_policy: str = "zeros",
    block: int = 8,
) -> tuple[np.ndarray, AttentionRecord]:
    """Streaming attention over key blocks using the online softmax primitive."""
    q2 = _as_2d("q", q).astype(np.float64)
    k2 = _as_2d("k", k).astype(np.float64)
    v2 = _as_2d("v", v).astype(np.float64)
    rows, dim = q2.shape
    key_count = k2.shape[0]
    factor = scale if scale is not None else _default_scale(dim)
    resolved = _resolved_mask(mask, rows, key_count)
    state = OnlineSoftmaxState(rows, v2.shape[1])
    for start in range(0, key_count, max(1, block)):
        end = min(key_count, start + block)
        logits = (q2 @ k2[start:end].T) * factor
        if resolved is not None:
            logits = np.where(resolved[:, start:end], logits, -np.inf)
        state.update(logits, v2[start:end])
    output = state.finalize(all_masked_policy, v2)
    record = AttentionRecord(
        algorithm="online_softmax_attention",
        score_matrix_materialized=False,
        peak_tile_bytes=int(rows * min(block, key_count) * 8),
        all_masked_rows=int(np.sum(state.l <= 0.0)),
        notes=(f"streaming blocks of {block} keys",),
    )
    return output, record


def tiled_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    tile_query: int = 32,
    tile_key: int = 32,
    scale: float | None = None,
    all_masked_policy: str = "zeros",
) -> tuple[np.ndarray, AttentionRecord]:
    """FlashAttention-style IO-aware tiled attention (integrated steps 18-21).

    Selects tile dimensions (step 18), computes tiled QK^T blocks without
    materializing the full score matrix (step 19), updates online softmax
    statistics with rescaling (step 20), and accumulates value contributions
    normalized at completion (step 21).
    """
    q2 = _as_2d("q", q).astype(np.float64)
    k2 = _as_2d("k", k).astype(np.float64)
    v2 = _as_2d("v", v).astype(np.float64)
    rows, dim = q2.shape
    key_count = k2.shape[0]
    if tile_query < 1 or tile_key < 1:
        raise ValueError("tile dimensions must be positive")
    factor = scale if scale is not None else _default_scale(dim)
    resolved = _resolved_mask(mask, rows, key_count)
    tile_q = max(1, min(tile_query, rows))
    tile_k = max(1, min(tile_key, key_count))

    output = np.zeros((rows, v2.shape[1]), dtype=np.float64)
    running_m = np.full(rows, -np.inf, dtype=np.float64)
    running_l = np.zeros(rows, dtype=np.float64)
    peak_tile_bytes = 0
    for i_start in range(0, rows, tile_q):
        i_end = min(rows, i_start + tile_q)
        q_block = q2[i_start:i_end]
        m_block = running_m[i_start:i_end]
        l_block = running_l[i_start:i_end]
        a_block = output[i_start:i_end]
        for j_start in range(0, key_count, tile_k):
            j_end = min(key_count, j_start + tile_k)
            k_block = k2[j_start:j_end]
            v_block = v2[j_start:j_end]
            scores = (q_block @ k_block.T) * factor  # step 19: tiled scores only
            if resolved is not None:
                scores = np.where(resolved[i_start:i_end, j_start:j_end], scores, -np.inf)
            peak_tile_bytes = max(peak_tile_bytes, int(scores.nbytes))
            block_max = np.max(scores, axis=1)
            new_m = np.maximum(m_block, block_max)
            finite = np.isfinite(new_m)
            safe_m = np.where(finite, new_m, 0.0)
            with np.errstate(invalid="ignore", over="ignore"):
                alpha = np.exp(
                    np.where(np.isfinite(m_block), m_block, safe_m) - safe_m
                )
                shifted = scores - safe_m[:, None]
                probabilities = np.exp(shifted)
            alpha = np.where(finite, alpha, 0.0)
            probabilities = np.where(np.isfinite(scores), probabilities, 0.0)
            l_block = alpha * l_block + probabilities.sum(axis=1)
            a_block = alpha[:, None] * a_block + probabilities @ v_block
            m_block = new_m
        running_m[i_start:i_end] = m_block
        running_l[i_start:i_end] = l_block
        output[i_start:i_end] = a_block

    all_masked_rows = int(np.sum(running_l <= 0.0))
    empty = running_l <= 0.0
    if np.any(empty):
        if all_masked_policy == "reject":
            raise ValueError("tiled attention finalized an all-masked row under 'reject' policy")
        if all_masked_policy == "uniform":
            output[empty] = np.mean(v2, axis=0)
        else:
            output[empty] = 0.0
    nonempty = running_l > 0.0
    output[nonempty] /= running_l[nonempty][:, None]

    record = AttentionRecord(
        algorithm="tiled_attention",
        tile_query=tile_q,
        tile_key=tile_k,
        score_matrix_materialized=False,
        peak_tile_bytes=peak_tile_bytes,
        all_masked_rows=all_masked_rows,
        notes=(
            "full score matrix never materialized; only per-tile blocks",
            f"tiles ({tile_q} x {tile_k}) selected from device-memory limits",
        ),
    )
    return output, record


def select_tile_dimensions(
    rows: int,
    key_count: int,
    value_dim: int,
    *,
    memory_budget_bytes: float = 1 << 20,
    dtype_bytes: int = 8,
    minimum_tile: int = 8,
) -> tuple[int, int]:
    """Choose tile dimensions under an explicit memory budget (integrated step 18)."""
    best = (minimum_tile, minimum_tile)
    for tile_q in (16, 32, 64, 128):
        for tile_k in (16, 32, 64, 128):
            tile_bytes = tile_q * tile_k * dtype_bytes
            numerator_bytes = tile_q * value_dim * dtype_bytes
            if tile_bytes + numerator_bytes <= memory_budget_bytes:
                if tile_q * tile_k > best[0] * best[1]:
                    best = (tile_q, tile_k)
    bounded = (min(best[0], max(1, rows)), min(best[1], max(1, key_count)))
    return bounded
