"""Diffusion-style generation benchmarked against an autoregressive baseline (Step 25).

Discrete diffusion corrupts, masks, and refines tokens rather than generating
strictly left-to-right.  The benchmark compares quality, editing capability,
parallelism, latency, and memory against an autoregressive baseline *for the actual
task* (here: restoring corrupted template sequences), and records the verdict
without assuming either family wins universally.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np

from ..nn import BigramLM, MaskDenoiserLM

__all__ = ("DiffusionBenchmark", "benchmark_diffusion_vs_autoregressive",
           "corrupt_sequences")


def corrupt_sequences(sequences: Sequence[Sequence[int]], mask_token: int,
                       corruption_rate: float, seed: int = 11
                       ) -> list[list[int]]:
    """Mask tokens at the given rate (the diffusion forward process)."""
    rng = np.random.default_rng(seed)
    corrupted = []
    for sequence in sequences:
        corrupted.append([int(token) if rng.random() > corruption_rate else mask_token
                          for token in sequence])
    return corrupted


@dataclass(slots=True)
class DiffusionBenchmark:
    """Step 25 -- measured comparison for the actual task."""

    task: str
    vocab_size: int
    sequence_length: int
    diffusion_quality: float
    autoregressive_quality: float
    diffusion_steps: int
    autoregressive_steps: int
    diffusion_latency_ms: float
    autoregressive_latency_ms: float
    diffusion_memory_bytes: int
    autoregressive_memory_bytes: int
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"task": self.task, "vocab_size": self.vocab_size,
                "sequence_length": self.sequence_length,
                "diffusion_quality": round(self.diffusion_quality, 6),
                "autoregressive_quality": round(self.autoregressive_quality, 6),
                "diffusion_steps": self.diffusion_steps,
                "autoregressive_steps": self.autoregressive_steps,
                "diffusion_latency_ms": round(self.diffusion_latency_ms, 3),
                "autoregressive_latency_ms": round(self.autoregressive_latency_ms, 3),
                "diffusion_memory_bytes": self.diffusion_memory_bytes,
                "autoregressive_memory_bytes": self.autoregressive_memory_bytes,
                "verdict": self.verdict,
                "note": "benchmark against a relevant baseline for the actual task; "
                        "no family is assumed to win universally"}


def benchmark_diffusion_vs_autoregressive(
        train_sequences: Sequence[Sequence[int]],
        test_sequences: Sequence[Sequence[int]], *,
        vocab_size: int, sequence_length: int, refinement_steps: int = 3,
        corruption_rate: float = 0.3,
        task: str = "restore corrupted template sequences") -> DiffusionBenchmark:
    """Train both families, corrupt the test set, and measure restoration quality.

    The diffusion model denoises masked positions in parallel (one pass per
    refinement step); the autoregressive baseline regenerates strictly
    left-to-right from a prompt.  Quality is exact-restoration accuracy; latency and
    memory are measured on the reference implementation.
    """
    if not train_sequences or not test_sequences:
        raise ValueError("both train and test sequences are required")
    denoiser = MaskDenoiserLM(vocab_size=vocab_size, sequence_length=sequence_length)
    denoiser.fit(train_sequences)
    bigram = BigramLM(vocab_size=vocab_size)
    bigram.fit(train_sequences)

    corrupted = corrupt_sequences(test_sequences, denoiser.mask_token,
                                  corruption_rate)

    started = time.perf_counter()
    restored = [denoiser.denoise(sequence) for sequence in corrupted]
    diffusion_latency = (time.perf_counter() - started) * 1000.0
    diffusion_quality = float(np.mean([
        int(restored_row == list(original))
        for restored_row, original in zip(restored, test_sequences)]))

    started = time.perf_counter()
    ar_restored = []
    for sequence in corrupted:
        # autoregressive: decode left-to-right from the first unmasked prompt token
        prompt_token = next((token for token in sequence if token != denoiser.mask_token),
                            0)
        generated = bigram.generate([prompt_token], sequence_length - 1)
        ar_restored.append(generated[-sequence_length:] if len(generated) >= sequence_length
                           else generated + [0] * (sequence_length - len(generated)))
    ar_latency = (time.perf_counter() - started) * 1000.0
    ar_quality = float(np.mean([
        int(restored_row == list(original))
        for restored_row, original in zip(ar_restored, test_sequences)]))

    if diffusion_quality > ar_quality:
        verdict = "diffusion wins on restoration quality for this task"
    elif ar_quality > diffusion_quality:
        verdict = "autoregressive baseline wins for this task"
    else:
        verdict = "tie on quality; decide on parallelism, latency and memory"
    return DiffusionBenchmark(
        task=task, vocab_size=vocab_size, sequence_length=sequence_length,
        diffusion_quality=diffusion_quality, autoregressive_quality=ar_quality,
        diffusion_steps=refinement_steps, autoregressive_steps=sequence_length - 1,
        diffusion_latency_ms=diffusion_latency,
        autoregressive_latency_ms=ar_latency,
        diffusion_memory_bytes=denoiser.memory_bytes(),
        autoregressive_memory_bytes=bigram.memory_bytes(),
        verdict=verdict)
