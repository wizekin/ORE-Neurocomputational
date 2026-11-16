"""Stage 6: convert evolving phonetic evidence into uncertain language-specific phoneme candidates."""
from __future__ import annotations

from typing import Any, Sequence
import numpy as np

from .contracts import PhonemeSpec, SensoryConfig, SensoryRequest
from .math_ops import sigmoid, softmax, safe_summary


_DEFAULT_PHONEMES = tuple(
    PhonemeSpec(symbol, manner, place, voiced, prior=1.0)
    for symbol, manner, place, voiced in (
        ("p","stop","bilabial",False),("b","stop","bilabial",True),
        ("t","stop","alveolar",False),("d","stop","alveolar",True),
        ("k","stop","velar",False),("g","stop","velar",True),
        ("f","fricative","labiodental",False),("v","fricative","labiodental",True),
        ("s","fricative","alveolar",False),("z","fricative","alveolar",True),
        ("ʃ","fricative","postalveolar",False),("ʒ","fricative","postalveolar",True),
        ("h","fricative","glottal",False),("m","nasal","bilabial",True),
        ("n","nasal","alveolar",True),("ŋ","nasal","velar",True),
        ("l","approximant","alveolar",True),("ɹ","approximant","postalveolar",True),
        ("j","approximant","palatal",True),("w","approximant","bilabial",True),
        ("tʃ","affricate","postalveolar",False),("dʒ","affricate","postalveolar",True),
    )
) + tuple(
    PhonemeSpec(symbol, "vowel", "unknown", True, height, backness, 1.0)
    for symbol, height, backness in (
        ("i","high","front"),("ɪ","near_high","front"),("e","mid","front"),
        ("ɛ","mid","front"),("æ","low","front"),("ə","mid","central"),
        ("ʌ","mid","central"),("ɑ","low","back"),("ɔ","mid","back"),
        ("o","mid","back"),("ʊ","near_high","back"),("u","high","back"),
    )
)
_MANNER_INDEX = {name: i for i, name in enumerate(("stop","fricative","nasal","approximant","affricate","vowel"))}
_PLACE_INDEX = {name: i for i, name in enumerate(("bilabial","labiodental","alveolar","postalveolar","palatal","velar","glottal","unknown"))}


def _safe_log(values: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Return bounded log probabilities for stable additive Bayesian evidence integration."""
    return np.log(np.maximum(np.asarray(values, dtype=np.float64), epsilon))


def _inventory(request: SensoryRequest) -> tuple[PhonemeSpec, ...]:
    """Use the caller's language inventory or a clearly marked English-oriented fallback inventory."""
    return tuple(request.phoneme_inventory) if request.phoneme_inventory else _DEFAULT_PHONEMES


class PhonemeIdentifier:
    """Implement Stage 6 with graded category populations, temporal evidence, context and recurrent settling.

    The default inventory is a demonstration and is not universal. A language/dialect-appropriate
    inventory must be supplied when the target speech system differs from the English-oriented fallback.
    """

    def __init__(self, config: SensoryConfig) -> None:
        """Store probability temperature, recurrent update and abstention thresholds for category decoding."""
        self.config = config

    def step_32(self, state: dict[str, Any]) -> dict[str, Any]:
        """Activate several language-specific phoneme categories in parallel from manner, place and voicing cues."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        inventory = _inventory(state["request"])
        state["phoneme_inventory_active"] = inventory
        manner, place = state["manner_posteriors"], state["place_posteriors"]
        voiced = np.clip(state["voicing_probability"], 1e-6, 1-1e-6)
        logits = []
        for item in inventory:
            mindex = _MANNER_INDEX.get(item.manner, _MANNER_INDEX["approximant"])
            score = _safe_log(manner[:, mindex])
            pindex = _PLACE_INDEX.get(item.place)
            if item.manner != "vowel" and pindex is not None:
                score += 0.65 * _safe_log(place[:, pindex])
            score += _safe_log(voiced if item.voiced else 1-voiced)
            logits.append(score)
        activation_logits = np.column_stack(logits)
        activations = softmax(activation_logits / self.config.phoneme_temperature, axis=1)
        symbols = tuple(item.symbol for item in inventory)
        state.update({"phoneme_activation_logits": activation_logits, "phoneme_activations": activations,
                      "phoneme_symbols": symbols})
        return {"candidate_count": len(inventory), "phoneme_symbols": list(symbols),
                "activation_summary": safe_summary(activations), "multiple_candidates_retained": True}

    def step_33(self, state: dict[str, Any]) -> dict[str, Any]:
        """Compare feature evidence with candidate categories and combine priors using Bayes-style log evidence."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        inventory: Sequence[PhonemeSpec] = state["phoneme_inventory_active"]
        evidence = state["phoneme_activation_logits"]
        prior = np.log(np.asarray([item.prior for item in inventory], dtype=np.float64))
        likelihood = softmax(evidence, axis=1)
        posterior = softmax(evidence + prior[None, :], axis=1)
        state.update({"category_likelihoods": likelihood, "category_priors": prior,
                      "category_posteriors": posterior})
        return {"likelihood_summary": safe_summary(likelihood), "posterior_summary": safe_summary(posterior),
                "prior_source": "caller inventory or documented fallback"}

    def step_34(self, state: dict[str, Any]) -> dict[str, Any]:
        """Accumulate frame-wise log evidence over time while retaining competing phoneme hypotheses."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        posterior = np.clip(state["category_posteriors"], 1e-10, 1.0)
        accumulated = np.cumsum(_safe_log(posterior), axis=0)
        time_scale = np.maximum(np.sqrt(np.arange(1, posterior.shape[0]+1))[:, None], 1.0)
        normalized = softmax(accumulated / time_scale, axis=1)
        state.update({"accumulated_log_evidence": accumulated, "temporally_integrated_posteriors": normalized})
        return {"accumulated_evidence": safe_summary(accumulated), "integrated_posteriors": safe_summary(normalized),
                "integration": "cumulative log evidence with square-root time normalization"}

    def step_35(self, state: dict[str, Any]) -> dict[str, Any]:
        """Apply a graded categorical-perception boundary to continuous voicing and voice-onset-time evidence."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        voiced = np.clip(state["voicing_probability"], 0, 1)
        vot = state.get("vot_seconds", float("nan"))
        boundary = 0.025
        vot_evidence = np.full(voiced.shape, 0.5) if not np.isfinite(vot) else sigmoid((boundary-vot)/0.012)
        signal_value = 0.65 * voiced + 0.35 * vot_evidence
        categorical = np.column_stack((1-signal_value, signal_value))
        state.update({"categorical_voicing_posteriors": categorical, "categorical_voicing_signal": signal_value})
        return {"voiceless_voiced_posteriors": safe_summary(categorical), "boundary_seconds": boundary,
                "boundary_model": "sigmoid((VOT - boundary) / 0.012)"}

    def step_36(self, state: dict[str, Any]) -> dict[str, Any]:
        """Bias ambiguous categories with bounded caller-provided phonetic/linguistic log-priors."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        posterior = np.clip(state["temporally_integrated_posteriors"], 1e-10, 1.0)
        context = state["request"].phonetic_context or {}
        symbols = state["phoneme_symbols"]
        prior = np.clip(np.asarray([float(context.get(symbol, 1.0)) for symbol in symbols]), 0.05, 20.0)
        contextual = softmax(_safe_log(posterior) + np.log(prior)[None, :], axis=1)
        state.update({"context_prior_vector": prior, "contextual_phoneme_posteriors": contextual})
        return {"contextual_posteriors": safe_summary(contextual), "context_used": bool(context),
                "context_symbols": list(context)}

    def step_37(self, state: dict[str, Any]) -> dict[str, Any]:
        """Iteratively integrate current acoustics, neighboring frames and contextual predictions with damped recurrence."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        evidence = state["contextual_phoneme_posteriors"]
        recurrent, history = evidence.copy(), []
        for _ in range(self.config.recurrent_iterations):
            previous = np.vstack((recurrent[:1], recurrent[:-1]))
            following = np.vstack((recurrent[1:], recurrent[-1:]))
            prediction = 0.5 * previous + 0.5 * following
            candidate = softmax(_safe_log(evidence) + 0.30 * _safe_log(prediction), axis=1)
            updated = (1-self.config.recurrent_damping)*recurrent + self.config.recurrent_damping*candidate
            updated /= np.maximum(np.sum(updated, axis=1, keepdims=True), 1e-12)
            history.append(float(np.mean(np.linalg.norm(updated-recurrent, axis=1))))
            recurrent = updated
        state.update({"recurrent_phoneme_state": recurrent, "recurrent_convergence_history": history})
        return {"recurrent_state": safe_summary(recurrent), "iterations": self.config.recurrent_iterations,
                "final_mean_update": history[-1] if history else 0.0}

    def step_38(self, state: dict[str, Any]) -> dict[str, Any]:
        """Select only above confidence and margin thresholds; report posterior entropy and abstain when uncertain."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        posterior = state["recurrent_phoneme_state"]
        order = np.argsort(posterior, axis=1)
        top = order[:, -1]
        best = posterior[np.arange(len(top)), top]
        second = posterior[np.arange(len(top)), order[:, -2]] if posterior.shape[1] > 1 else np.zeros_like(best)
        margin = best-second
        entropy = -np.sum(posterior*_safe_log(posterior), axis=1)
        symbols = state["phoneme_symbols"]
        selected = [symbols[i] if best[row] >= self.config.selection_confidence and
                    margin[row] >= self.config.selection_margin else None for row, i in enumerate(top)]
        state.update({"selected_phoneme_frames": selected, "candidate_probabilities": posterior,
                      "phoneme_frame_confidence": best, "phoneme_frame_margin": margin, "phoneme_entropy": entropy})
        return {"selected_frame_count": sum(x is not None for x in selected), "frame_count": len(selected),
                "mean_confidence": float(np.mean(best)), "mean_margin": float(np.mean(margin)),
                "mean_entropy_nats": float(np.mean(entropy)),
                "uncertain_frame_count": sum(x is None for x in selected)}

    def step_39(self, state: dict[str, Any]) -> dict[str, Any]:
        """Construct ordered phoneme-segment hypotheses with frame-derived onset, offset, duration and confidence."""
        if not state.get("auditory_available"):
            return {"status": "skipped_no_auditory_input"}
        labels, posterior, times = state["selected_phoneme_frames"], state["candidate_probabilities"], state["stft_times_seconds"]
        hop = float(np.median(np.diff(times))) if times.size > 1 else self.config.auditory_hop_ms/1000
        sequence, start = [], 0
        while start < len(labels):
            label, end = labels[start], start+1
            while end < len(labels) and labels[end] == label:
                end += 1
            if label is not None:
                mean_p = np.mean(posterior[start:end], axis=0)
                index = int(np.argmax(mean_p))
                sequence.append({"symbol": label, "onset_seconds": float(times[start]),
                                 "offset_seconds": float(times[min(end-1, times.size-1)]+hop),
                                 "duration_seconds": float((end-start)*hop),
                                 "confidence": float(mean_p[index]),
                                 "uncertainty_nats": float(np.mean(state["phoneme_entropy"][start:end])),
                                 "frame_span": [start, end]})
            start = end
        state["phoneme_sequence"] = sequence
        state["sequence_confidence"] = float(np.mean([x["confidence"] for x in sequence])) if sequence else 0.0
        return {"phoneme_sequence": sequence, "sequence_confidence": state["sequence_confidence"],
                "segment_count": len(sequence),
                "boundary_note": "frame-derived estimates, not exact articulatory boundaries"}
