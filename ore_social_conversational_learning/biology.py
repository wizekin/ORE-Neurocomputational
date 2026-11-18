"""Distributed biological-function surrogates for Point 14 social and conversational learning.

These objects model functional contributions of interacting systems; they are not anatomical replicas,
direct readouts of brain activity, or claims that a single region performs an entire operation.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import (
    LearnerSocialState, PartnerModel, SocialInteractionEvent, SocialLearningConfig,
)
from .functional import (
    design_audience_response, estimate_social_attention,
    evaluate_explanation, infer_communicative_intention,
    infer_joint_attention, integrate_multimodal_signals,
    predict_turn_state, select_scaffold,
)


class SuperiorTemporalSocialPerceptionNetwork:
    """Model extraction of socially relevant voice, gaze, gesture, facial, and action cues."""

    def __init__(self) -> None:
        """Initialize social-perception activity and retained cue diagnostics."""
        self.processing_count = 0
        self.last_modalities: tuple[str, ...] = ()

    def process(
        self,
        event: SocialInteractionEvent,
        config: SocialLearningConfig,
    ) -> dict[str, Any]:
        """Prioritize social cues and integrate their modality-specific content."""
        self.processing_count += 1
        attention = estimate_social_attention(event.signals, event.learning_goal, config)
        self.last_modalities = tuple(sorted({signal.modality for signal in event.signals}))
        return {
            "attention": attention,
            "modalities": self.last_modalities,
            "signal_count": len(event.signals),
        }

    def snapshot(self) -> dict[str, Any]:
        """Expose sensory-social processing diagnostics without making a mind-reading claim."""
        return {
            "model": "superior_temporal_social_perception_surrogate",
            "processing_count": self.processing_count,
            "last_modalities": self.last_modalities,
        }


class TemporoparietalJointAttentionNetwork:
    """Model partner-focus estimation and shared-referent coordination."""

    def __init__(self) -> None:
        """Initialize joint-attention monitoring counters and confidence state."""
        self.update_count = 0
        self.last_confidence = 0.0
        self.last_status = "unknown"

    def update(
        self,
        event: SocialInteractionEvent,
        attention: Mapping[str, Any],
        config: SocialLearningConfig,
    ) -> dict[str, Any]:
        """Estimate whether learner and partner are jointly attending to one referent."""
        self.update_count += 1
        result = infer_joint_attention(event, attention, config)
        self.last_confidence = float(result["shared_attention_confidence"])
        self.last_status = str(result["status"])
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return the shared-attention state and evidence-based uncertainty marker."""
        return {
            "model": "temporoparietal_joint_attention_surrogate",
            "update_count": self.update_count,
            "last_confidence": self.last_confidence,
            "last_status": self.last_status,
        }


class MedialPrefrontalMentalizingNetwork:
    """Model probabilistic inference about partner goals, knowledge, and perspective."""

    def __init__(self) -> None:
        """Initialize partner-intention inference counters and uncertainty diagnostics."""
        self.inference_count = 0
        self.last_intention = ""
        self.last_uncertainty = 0.0

    def infer(
        self,
        event: SocialInteractionEvent,
        partner: PartnerModel,
        config: SocialLearningConfig,
    ) -> dict[str, Any]:
        """Infer communicative intention and partner-knowledge gaps without asserting certainty."""
        self.inference_count += 1
        result = infer_communicative_intention(event, partner, config)
        self.last_intention = str(result["most_likely_intention"])
        self.last_uncertainty = float(result["uncertainty_nats"])
        return result

    def snapshot(self) -> dict[str, Any]:
        """Expose mentalizing-style inference activity and retained uncertainty."""
        return {
            "model": "medial_prefrontal_mentalizing_surrogate",
            "inference_count": self.inference_count,
            "last_intention": self.last_intention,
            "last_uncertainty_nats": self.last_uncertainty,
        }


class FrontotemporalLanguageInteractionNetwork:
    """Model interactions between linguistic content, multimodal evidence, and grounded semantics."""

    def __init__(self) -> None:
        """Initialize speech integration and semantic-grounding diagnostics."""
        self.integration_count = 0
        self.grounding_count = 0
        self.last_modalities: tuple[str, ...] = ()

    def integrate(
        self,
        event: SocialInteractionEvent,
        attention: Mapping[str, Any],
        config: SocialLearningConfig,
    ) -> dict[str, Any]:
        """Combine speech, prosody, face, gesture, gaze, and action cues with source tags."""
        self.integration_count += 1
        result = integrate_multimodal_signals(event, attention, config)
        self.last_modalities = tuple(sorted(result["modality_weights"]))
        return result

    def record_grounding(self) -> None:
        """Record the completion of a word-to-scene grounding operation."""
        self.grounding_count += 1

    def snapshot(self) -> dict[str, Any]:
        """Return language/social-integration activity and supported modalities."""
        return {
            "model": "frontotemporal_multimodal_language_interaction_surrogate",
            "integration_count": self.integration_count,
            "grounding_count": self.grounding_count,
            "last_modalities": self.last_modalities,
        }


class InferiorFrontalTurnCoordinationNetwork:
    """Model turn-state prediction and response readiness using timing and linguistic cues."""

    def __init__(self) -> None:
        """Initialize conversational turn prediction counters."""
        self.prediction_count = 0
        self.last_state = "unknown"

    def predict(
        self,
        event: SocialInteractionEvent,
        attention: Mapping[str, Any],
        config: SocialLearningConfig,
    ) -> dict[str, Any]:
        """Estimate whether the partner continues, yields, or signals a repair."""
        self.prediction_count += 1
        result = predict_turn_state(event, attention, config)
        self.last_state = str(result["predicted_state"])
        return result

    def snapshot(self) -> dict[str, Any]:
        """Expose turn-state prediction activity."""
        return {
            "model": "inferior_frontal_auditory_motor_turn_coordination_surrogate",
            "prediction_count": self.prediction_count,
            "last_state": self.last_state,
        }


class PrefrontalScaffoldingControlNetwork:
    """Model goal maintenance, audience design, adaptive support, and fading of assistance."""

    def __init__(self) -> None:
        """Initialize control and scaffolding operation counters."""
        self.audience_design_count = 0
        self.scaffold_count = 0
        self.last_support_level = 0.0

    def design_response(
        self,
        event: SocialInteractionEvent,
        partner: PartnerModel,
        intention: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Adapt response vocabulary and detail to the estimated partner knowledge state."""
        self.audience_design_count += 1
        return design_audience_response(event, partner, intention)

    def scaffold(
        self,
        event: SocialInteractionEvent,
        state: LearnerSocialState,
        config: SocialLearningConfig,
    ) -> dict[str, Any]:
        """Select a minimal useful hint, example, question, or independent retrieval prompt."""
        self.scaffold_count += 1
        result = select_scaffold(event, state, config)
        self.last_support_level = float(result["support_level"])
        return result

    def snapshot(self) -> dict[str, Any]:
        """Return audience-design and scaffold-strength diagnostics."""
        return {
            "model": "prefrontal_adaptive_scaffolding_surrogate",
            "audience_design_count": self.audience_design_count,
            "scaffold_count": self.scaffold_count,
            "last_support_level": self.last_support_level,
        }


class VentralStriatalSocialRewardNetwork:
    """Model reward learning from social outcomes separately from factual belief revision."""

    def __init__(self) -> None:
        """Initialize social reward update counters and value diagnostics."""
        self.update_count = 0
        self.last_prediction_error = 0.0

    def update(self, prediction_errors: Sequence[float]) -> dict[str, Any]:
        """Aggregate social/outcome prediction errors without treating praise as factual proof."""
        self.update_count += 1
        self.last_prediction_error = (
            sum(float(item) for item in prediction_errors) / max(1, len(prediction_errors))
        )
        return {
            "mean_social_outcome_prediction_error": self.last_prediction_error,
            "feedback_count": len(prediction_errors),
            "factual_truth_channel_separate": True,
        }

    def snapshot(self) -> dict[str, Any]:
        """Expose social-outcome learning diagnostics and its scope limitation."""
        return {
            "model": "ventral_striatal_social_reward_surrogate",
            "update_count": self.update_count,
            "last_prediction_error": self.last_prediction_error,
            "scope": "social and task outcome value, not factual truth evaluation",
        }


class HippocampalSocialEpisodicNetwork:
    """Model binding of concept content, partner source, interaction context, and feedback."""

    def __init__(self) -> None:
        """Initialize source-linked episodic encoding and retrieval counters."""
        self.encoding_count = 0
        self.retrieval_count = 0
        self.last_event_id = ""

    def encode(self, event_id: str) -> None:
        """Record creation of one source-linked interaction episode."""
        self.encoding_count += 1
        self.last_event_id = event_id

    def retrieved(self, count: int) -> None:
        """Record how many earlier social episodes were reactivated by cues."""
        self.retrieval_count += max(0, int(count))

    def snapshot(self) -> dict[str, Any]:
        """Expose episodic source-binding activity."""
        return {
            "model": "hippocampal_social_episode_source_binding_surrogate",
            "encoding_count": self.encoding_count,
            "retrieved_episode_count": self.retrieval_count,
            "last_event_id": self.last_event_id,
        }


class PosteriorMedialNarrativeNetwork:
    """Model temporal, causal, and agent-centered integration of dialogue and story events."""

    def __init__(self) -> None:
        """Initialize narrative event-model activity counters."""
        self.integration_count = 0
        self.last_node_count = 0

    def integrate(self, narrative: Mapping[str, Any]) -> dict[str, Any]:
        """Register construction of an event graph and agent-specific knowledge model."""
        self.integration_count += 1
        self.last_node_count = int(narrative.get("graph_node_count", 0))
        return {
            "integration_count": self.integration_count,
            "narrative_node_count": self.last_node_count,
            "agent_states_tracked": len(narrative.get("agent_specific_states", {})),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return narrative-event integration diagnostics."""
        return {
            "model": "posterior_medial_narrative_event_surrogate",
            "integration_count": self.integration_count,
            "last_node_count": self.last_node_count,
        }


class BiologicalSocialLearningNetwork:
    """Coordinate distributed social-perception, language, mentalizing, control, reward, and memory surrogates."""

    def __init__(self) -> None:
        """Initialize interacting functional surrogates for Point 14."""
        self.superior_temporal = SuperiorTemporalSocialPerceptionNetwork()
        self.temporoparietal_joint_attention = TemporoparietalJointAttentionNetwork()
        self.medial_prefrontal_mentalizing = MedialPrefrontalMentalizingNetwork()
        self.frontotemporal_language = FrontotemporalLanguageInteractionNetwork()
        self.inferior_frontal_turns = InferiorFrontalTurnCoordinationNetwork()
        self.prefrontal_control = PrefrontalScaffoldingControlNetwork()
        self.ventral_striatal_reward = VentralStriatalSocialRewardNetwork()
        self.hippocampal_memory = HippocampalSocialEpisodicNetwork()
        self.posterior_medial_narrative = PosteriorMedialNarrativeNetwork()

    def snapshot(self) -> dict[str, Any]:
        """Return all social-learning subsystem diagnostics and biological-model limitations."""
        return {
            "superior_temporal": self.superior_temporal.snapshot(),
            "temporoparietal_joint_attention": self.temporoparietal_joint_attention.snapshot(),
            "medial_prefrontal_mentalizing": self.medial_prefrontal_mentalizing.snapshot(),
            "frontotemporal_language": self.frontotemporal_language.snapshot(),
            "inferior_frontal_turns": self.inferior_frontal_turns.snapshot(),
            "prefrontal_control": self.prefrontal_control.snapshot(),
            "ventral_striatal_reward": self.ventral_striatal_reward.snapshot(),
            "hippocampal_memory": self.hippocampal_memory.snapshot(),
            "posterior_medial_narrative": self.posterior_medial_narrative.snapshot(),
            "scientific_scope": (
                "Distributed functional surrogates; not an anatomical connectome, literal brain replica, "
                "direct brain-to-brain transfer, or a claim that synchrony proves shared understanding."
            ),
        }
