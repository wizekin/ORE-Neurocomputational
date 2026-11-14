"""Executable 40-operation registry for the seven-stage predictive ORE algorithm."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class OperationSpec:
    operation_id: str
    stage_id: str
    name: str
    mathematical_form: str
    biological_reference: str
    implementation_source: str

@dataclass(frozen=True, slots=True)
class StageSpec:
    stage_id: str
    name: str
    purpose: str
    operation_ids: tuple[str, ...]

_OPERATION_ROWS = (
    ("S1", "Validate supervised grapheme references", "L={(x_j,c_j)}; c_j in C",
     "Validate sensory teaching signals and stable feature-population labels.", "ore_n3.pipeline.N3Processor.fit"),
    ("S1", "Extract visual-structure descriptors", "x_j=Descriptor(N2_j)",
     "Pool contour, boundary, junction, curvature, figure-ground and shape-graph signals.", "ore_n3.visual.extract_patch_features"),
    ("S1", "Estimate feature centering and scale", "mu=mean(x); sigma=max(std(x),epsilon)",
     "Estimate population baseline and feature-wise gain normalization.", "ore_n3.visual.VisualPopulation.fit"),
    ("S1", "Standardize and normalize reference patterns", "z_j=normalize((x_j-mu)/sigma)",
     "Normalize distributed feature activity before forming class-selective patterns.", "ore_n3.visual.VisualPopulation.fit"),
    ("S1", "Construct grapheme population prototypes", "w_c=normalize(mean({z_j:c_j=c}))",
     "Represent learned grapheme-selective population templates; these are computational prototypes.", "ore_n3.visual.VisualPopulation.fit"),
    ("S1", "Compute query-to-prototype similarity", "a_i(c)=z_i dot w_c",
     "Read out distributed visual evidence across candidate grapheme populations.", "ore_n3.visual.VisualPopulation.probabilities"),
    ("S1", "Convert similarity to candidate activation", "p_i(c|known)=softmax(a_i(c)/T)",
     "Use graded competitive activity across known grapheme-selective populations.", "ore_n3.visual.VisualPopulation.probabilities"),
    ("S1", "Preserve uncertainty as an unknown candidate", "q_i(c)=r_i p_i(c|known); q_i(UNK)=1-r_i",
     "Retain residual activity for uncertain or unlearned visual patterns instead of forcing a known class.", "ore_n3.visual.VisualPopulation.probabilities"),

    ("S2", "Count grapheme exposures", "N(c)=sum_j 1[c_j=c]",
     "Accumulate experience-dependent frequency traces for grapheme populations.", "ore_n3.transitions.OrthographicTransitionModel.fit"),
    ("S2", "Count ordered adjacent grapheme pairs", "N(a,b)=sum_t 1[c_t=a,c_(t+1)=b]",
     "Learn direction-sensitive co-activation across neighboring grapheme positions.", "ore_n3.transitions.OrthographicTransitionModel.fit"),
    ("S2", "Apply additive Dirichlet smoothing", "N_alpha(c)=N(c)+alpha; N_alpha(a,b)=N(a,b)+alpha",
     "Avoid zero prior or transition activity for unseen but possible combinations.", "ore_n3.transitions.OrthographicTransitionModel.fit"),
    ("S2", "Estimate the smoothed unigram prior", "P(c)=N_alpha(c)/sum_k N_alpha(k)",
     "Maintain a baseline population prior when contextual neighbors are absent.", "ore_n3.transitions.OrthographicTransitionModel.fit"),
    ("S2", "Normalize forward transitions", "P(b|a)=N_alpha(a,b)/sum_j N_alpha(a,j)",
     "Encode forward conditional associations from a preceding grapheme to a following one.", "ore_n3.transitions.OrthographicTransitionModel.fit"),
    ("S2", "Normalize reverse transitions", "P(a|b)=N_alpha(a,b)/sum_i N_alpha(i,b)",
     "Encode the reverse conditional view without reversing the stored sequence order.", "ore_n3.transitions.OrthographicTransitionModel.fit"),
    ("S2", "Freeze and fingerprint transition memory", "h=SHA256(C,P(c),P(b|a),P(a|b))",
     "Keep learned association strengths stable during recognition; learning is a separate action.", "ore_n3.transitions.OrthographicTransitionModel.fit"),

    ("S3", "Initialize grapheme beliefs from visual evidence", "b_i^(0)=q_i",
     "Initialize recurrent candidate populations from bottom-up sensory evidence.", "ore_n3.pipeline.N3Processor.recognize"),
    ("S3", "Generate left-neighbor predictions", "m_i^L=b_(i-1) P(next|current)",
     "Propagate predictive excitation from the preceding grapheme position.", "ore_n3.transitions.OrthographicTransitionModel.predict_from_left"),
    ("S3", "Generate right-neighbor predictions", "m_i^R=P(previous|next) b_(i+1)",
     "Propagate predictive excitation from the following grapheme position.", "ore_n3.transitions.OrthographicTransitionModel.predict_from_right"),
    ("S3", "Combine available neighbor messages and unigram prior", "u_i=w_u u+w_L m_i^L+w_R m_i^R",
     "Integrate recurrent messages while omitting unavailable boundary messages.", "ore_n3.stages.contextual_prior"),
    ("S3", "Normalize the contextual prediction", "p_i=u_i/sum_c u_i(c)",
     "Convert combined synaptic support into a normalized candidate prior.", "ore_n3.stages.contextual_prior"),

    ("S4", "Measure visual uncertainty", "H_i=-sum_c q_i(c) log(q_i(c)+epsilon)",
     "Estimate uncertainty from the distribution of active candidate populations.", "ore_n4.stages.normalized_entropy"),
    ("S4", "Normalize uncertainty across class count", "U_i=H_i/log(max(|C|,2))",
     "Scale uncertainty to a comparable bounded gain-control signal.", "ore_n4.stages.normalized_entropy"),
    ("S4", "Reduce classwise mismatch to a position signal", "E_i=scale(mean_c |e_i(c)|)",
     "Pool local prediction-error activity into a bottom-up attentional drive.", "ore_n4.pipeline.N4Processor.process"),
    ("S4", "Measure visual-context disagreement", "D_i=(KL(q_i||p_i)+KL(p_i||q_i))/2",
     "Use mismatch between sensory and predicted candidate activity as a salience cue.", "ore_n4.stages.symmetric_kl_rows"),
    ("S4", "Construct bottom-up salience", "s_i=clip(w_U U_i+w_E E_i+w_D scale(D_i),0,1)",
     "Combine uncertainty, error and surprise into a bounded attentional signal.", "ore_n4.pipeline.N4Processor.process"),
    ("S4", "Integrate task, class-goal and seed relevance", "t_i=clip(w_T T_i+w_G G_i+w_A A_i,0,1)",
     "Allow goal/task signals to bias attention without fabricating absent goals.", "ore_n4.pipeline.N4Processor.process"),
    ("S4", "Allocate and smooth competitive attention", "a=normalize(smooth((1-lambda)softmax((s+t)/T)+lambda/N))",
     "Model competitive spatial selection, a nonzero coverage floor and local spread.", "ore_n4.stages.smooth_spatial_attention"),
    ("S4", "Compute attention-dependent sensory precision", "pi_s=clip(pi_0(0.5+0.5r)(1+g_a N a),pi_min,pi_max)",
     "Represent attention as a gain on the reliability of incoming sensory evidence.", "ore_n4.pipeline.N4Processor.process"),

    ("S5", "Estimate prior precision from uncertainty and reliability", "pi_p=g_p(0.5+U_i)(2-r_i)",
     "Increase reliance on predictions when sensory evidence is uncertain or unreliable.", "ore_n4.pipeline.N4Processor.process"),
    ("S5", "Modulate prior precision with top-down relevance", "pi_p=pi_p(1+g_t t_i)",
     "Represent task-dependent gain on predictive signals.", "ore_n4.pipeline.N4Processor.process"),
    ("S5", "Bound prior precision", "pi_p=clip(pi_p, pi_p_min, pi_p_max)",
     "Prevent unstable or effectively infinite predictive gain.", "ore_n4.pipeline.N4Processor.process"),
    ("S5", "Validate separate sensory and prior precision channels", "finite(pi_s,pi_p) and pi_s,pi_p>0",
     "Keep feedforward sensory gain distinct from top-down predictive gain.", "ore_n4.pipeline.N4Processor.process"),

    ("S6", "Initialize attended belief state", "b_i^(0)=b_i^N3",
     "Seed a new recurrent state while leaving the upstream N3 state unchanged.", "ore_n4.pipeline.N4Processor.process"),
    ("S6", "Form precision-weighted log-evidence", "l_i(c)=pi_s g_i(c)log(q_i(c))+pi_p log(p_i(c))",
     "Combine precision-scaled sensory and predictive synaptic evidence.", "ore_n4.stages.damped_belief_update"),
    ("S6", "Normalize the posterior proposal", "qhat_i=softmax(l_i)",
     "Convert integrated candidate drive into a normalized posterior proposal.", "ore_n4.stages.damped_belief_update"),
    ("S6", "Apply damped recurrent posterior update", "b_i^(k+1)=(1-rho)b_i^k+rho qhat_i",
     "Represent gradual recurrent settling rather than an instantaneous winner-take-all jump.", "ore_n4.stages.damped_belief_update"),
    ("S6", "Stop on convergence or iteration limit", "mean(|b^(k+1)-b^k|)<=tau",
     "Use bounded settling and a measurable stopping criterion for recurrent activity.", "ore_n4.pipeline.N4Processor.process"),

    ("S7", "Calculate posterior-prediction divergence", "e_i(c)=b_i(c)(log b_i(c)-log p_i(c))",
     "Represent candidate-specific mismatch and retain the full error distribution.", "ore_n4.pipeline.N4Processor.process"),
    ("S7", "Calculate feature-space prediction residuals", "r_i=z_i-sum_c p_i(c)w_c; r_i^att=z_i-sum_c b_i(c)w_c",
     "Compare observed distributed visual features with context- and posterior-predicted feature patterns.", "ore_predictive_orthography.pipeline.PredictiveOrthographyProcessor"),
    ("S7", "Validate, diagnose, rank and preserve alternatives", "diagnostics={KL,||r||,Delta b,energy proxy,confidence,margin}",
     "Read out uncertainty and mismatch while retaining alternatives or abstaining when evidence is weak.", "ore_predictive_orthography.pipeline.PredictiveOrthographyProcessor"),
)

_OPERATION_STAGE_COUNTS = {"S1": 8, "S2": 7, "S3": 5, "S4": 8, "S5": 4, "S6": 5, "S7": 3}
_STAGE_INFO = (
    ("S1", "Bottom-up character evidence", "Convert labelled, already-segmented visual patches into graded grapheme beliefs."),
    ("S2", "Orthographic transition probabilities", "Learn smoothed forward and reverse conditional statistics from labelled grapheme sequences."),
    ("S3", "Predictions from neighboring grapheme beliefs", "Exchange left- and right-neighbor messages to construct contextual candidate priors."),
    ("S4", "Attention-dependent sensory precision", "Allocate position- and class-selective attention and use it to modulate sensory reliability."),
    ("S5", "Prior precision modulation", "Adjust top-down predictive gain independently from sensory precision."),
    ("S6", "Iterative posterior updates", "Reconcile sensory and predictive evidence through bounded damped belief updates."),
    ("S7", "Prediction divergence and feature-space residual diagnostics", "Measure candidate and feature-space mismatch, validate the output and preserve alternatives."),
)

MASTER_OPERATION_REGISTRY: tuple[OperationSpec, ...] = tuple(
    OperationSpec(
        operation_id=f"MPO-{index:03d}",
        stage_id=row[0],
        name=row[1],
        mathematical_form=row[2],
        biological_reference=row[3],
        implementation_source=row[4],
    )
    for index, row in enumerate(_OPERATION_ROWS, start=1)
)
_STAGE_IDS = {stage_id: [] for stage_id, _, _ in _STAGE_INFO}
for _operation in MASTER_OPERATION_REGISTRY:
    _STAGE_IDS[_operation.stage_id].append(_operation.operation_id)
MASTER_STAGE_REGISTRY: tuple[StageSpec, ...] = tuple(
    StageSpec(stage_id, name, purpose, tuple(_STAGE_IDS[stage_id]))
    for stage_id, name, purpose in _STAGE_INFO
)
if len(MASTER_OPERATION_REGISTRY) != 40 or len(MASTER_STAGE_REGISTRY) != 7:
    raise RuntimeError("the master predictive-orthography registry must contain 7 stages and 40 operations")
if any(len(stage.operation_ids) != _OPERATION_STAGE_COUNTS[stage.stage_id] for stage in MASTER_STAGE_REGISTRY):
    raise RuntimeError("master registry stage operation counts are inconsistent")

def registry_as_dict() -> dict[str, object]:
    """Return JSON-friendly, stable registry metadata."""
    return {
        "stages": [
            {"stage_id": s.stage_id, "name": s.name, "purpose": s.purpose,
             "operation_ids": list(s.operation_ids)}
            for s in MASTER_STAGE_REGISTRY
        ],
        "operations": [
            {"operation_id": o.operation_id, "stage_id": o.stage_id, "name": o.name,
             "mathematical_form": o.mathematical_form,
             "biological_reference": o.biological_reference,
             "implementation_source": o.implementation_source}
            for o in MASTER_OPERATION_REGISTRY
        ],
    }
