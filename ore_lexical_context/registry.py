"""Formal 40-operation execution registry for lexical-context cognition."""
from __future__ import annotations
from dataclasses import asdict, dataclass

@dataclass(frozen=True, slots=True)
class OperationSpec:
    operation_id: str
    subsystem: str
    name: str
    functional_responsibility: str
    mathematical_form: str
    biological_reference: str
    def to_dict(self): return asdict(self)

_ROWS = (
("Lexical activation","Retrieve accumulated word exposure","Read corpus frequency and previous encounter counts from the entry and session state.","n_w = n_lexicon(w) + n_session(w)","Long-term lexical familiarity and recent exposure can bias baseline response; counts are software proxies."),
("Lexical activation","Calculate logarithmic frequency strength","Compress skewed word-exposure counts into a bounded prior activation.","F_w = log(1+n_w) / log(1+n_max)","Frequency-sensitive populations may show graded differences in resting excitability."),
("Lexical activation","Construct lexical resting activation","Set a pre-stimulus activation baseline from frequency and tonic bias.","R_w = b_0 + alpha_F F_w","Resting activation approximates baseline readiness of word-form populations."),
("Lexical activation","Adjust lexical activation thresholds","Lower thresholds for familiar forms and raise them with accumulated adaptation.","theta_w = theta_0 - alpha_theta F_w + beta_A A_w","Threshold modulation and adaptation are plausible population-level influences."),
("Lexical activation","Generate frequency-conditioned activation","Combine bottom-up match, baseline, threshold, repetition and adaptation.","a_w = sigmoid((R_w + g_s S_w + g_r H_w - theta_w - g_A A_w)/T)","Sensory and top-down input alter lexical population response probability/rate."),
("Lexical activation","Retrieve recent lexical activation history","Retrieve prior activation events and their elapsed times for this form.","H_w = {(a_k, t_k) : k < t}","Recent activation traces may be reinstated by recurrent/synaptic state."),
("Lexical activation","Calculate stimulus similarity","Estimate orthographic match between the input and each lexical entry.","S_w = 1 - d_Lev(input,w)/max(|input|,|w|)","Distributed letter/position populations provide graded bottom-up match."),
("Lexical activation","Calculate temporal decay","Decay prior activation as a function of time since the last encounter.","D_w = exp(-Delta_t/tau_H)","Short-term activity and synaptic traces decay over time constants."),
("Lexical activation","Generate repetition-dependent facilitation","Increase activation for repeated forms with saturating dependence on recent count.","P_w = g_P log(1+n_recent)/log(1+n_cap) * D_w","Repetition priming may facilitate subsequent responses; spacing changes its magnitude."),
("Lexical activation","Apply repetition-related neural adaptation","Track a decaying fatigue/adaptation state and subtract it from current activation.","A_w(t)=exp(-Delta_t/tau_A)A_w(t-1)+eta*a_w(t-1)","Activity-dependent adaptation and refractoriness reduce repeated population response."),
("Morphological analysis","Detect possible morphological boundaries","Generate candidate prefix/root/suffix decompositions from the supplied morpheme inventory.","B = {(prefix,stem,suffix): input = prefix + stem + suffix}","Morphological processing may recruit partly decomposable form representations."),
("Morphological analysis","Evaluate stem and affix candidates","Score stems by lexical support and affixes by productivity, frequency and fit.","M = F_stem * product(P_affix * F_affix)","Stem and affix evidence jointly contribute to morphologically structured activation."),
("Morphological analysis","Retrieve morpheme-associated meanings","Retrieve only supplied semantic feature maps for candidate stems and affixes.","mu_m = feature_map(morpheme)","Morpheme-linked meaning is a lexical-semantic association, not a universal decomposition rule."),
("Morphological analysis","Evaluate morphological composition","Compose stem and affix features while preserving contribution provenance.","mu_comp(f)=clip(mu_stem(f)+sum_j eta_j*mu_affix_j(f),0,1)","Distributed semantic contributions can combine during word interpretation."),
("Morphological analysis","Integrate whole-word and decomposed representations","Blend lexical whole-form evidence and supported morphological composition.","mu_w=q*mu_whole+(1-q)*mu_comp","Whole-form and decomposed routes may coactivate and converge."),
("Phonological mapping","Generate grapheme-to-phoneme correspondences","Produce lexical and rule-based pronunciation candidates from known entries/rules.","G2P(w)={phi_lex,phi_rule,...}","Orthographic and phonological populations exchange learned mapping signals."),
("Phonological mapping","Calculate mapping consistency","Measure grapheme coverage and reliability of rule sequences for each mapping.","C_phi=coverage*mean(rule_reliability)","Consistent grapheme-phoneme correspondences provide stronger phonological evidence."),
("Phonological mapping","Activate competing pronunciations","Preserve distinct lexical and sublexical phoneme sequences as alternatives.","Phi={phi_1,...,phi_k}","Competing phonological representations can be simultaneously active."),
("Phonological mapping","Weight pronunciation alternatives","Weight alternatives by lexical prior, mapping reliability and form similarity.","P(phi_i|w)=softmax(log prior_i+log reliability_i+beta*C_i)","Competition between pronunciation populations reflects differential evidence."),
("Phonological mapping","Send phonological evidence into lexical recognition","Use pronunciation compatibility to refine lexical activation without erasing orthographic evidence.","a'_w=clip(a_w+g_phi*sim(phi_w,phi_input),0,1)","Bidirectional orthographic-phonological interactions can refine lexical candidates."),
("Orthographic competition","Generate orthographic neighbors","Retrieve entries within the configured edit-distance neighborhood.","N(w)={v:d_Lev(w,v)<=d_max,v!=w}","Neighboring word-form populations can coactivate from shared letter evidence."),
("Orthographic competition","Calculate neighborhood strength","Estimate frequency-weighted activation in the input's orthographic neighborhood.","NS_w=sum_{v in N(w)} F_v*S_v/(|N(w)|+epsilon)","Neighborhood density and frequency can modulate lexical competition."),
("Orthographic competition","Calculate shared orthographic activation","Measure shared character n-grams and pool activation from overlapping candidates.","SO_w=sum_v Dice(ngrams(w,v))*a_v","Overlapping letter detectors can support shared activation across word forms."),
("Orthographic competition","Apply competitive inhibition","Subtract bounded similarity-weighted neighbor drive while preserving raw evidence.","a_inh_w=max(0,a_w-lambda*sum_v sim(w,v)*a_v)","Local inhibitory interactions may reduce coactive competing populations."),
("Orthographic competition","Settle lexical competition","Run a damped recurrent competition loop and retain ranked alternatives.","a(t+1)=(1-rho)a(t)+rho*sigmoid(u+g_e*a-g_i*W*a)","Recurrent excitation and lateral inhibition can stabilize lexical population patterns."),
("Semantic ambiguity","Detect semantic ambiguity","Enumerate multiple supplied meaning hypotheses for active lexical candidates.","K_w = number of supported meaning hypotheses","Multiple semantic representations may be coactive for ambiguous forms."),
("Semantic ambiguity","Generate prior probabilities for meanings","Normalize meaning priors using sense frequency and lexical candidate evidence.","P(m)=prior_m*lexical_activation_m / sum_j(prior_j*activation_j)","Meaning frequency and lexical access bias initial semantic activation."),
("Semantic ambiguity","Retrieve meaning-specific features","Load the provenance-bearing feature map for each meaning alternative.","mu_m={f:x_f} from supplied meaning inventory","Meaning-specific feature populations contribute to semantic representation."),
("Semantic ambiguity","Calculate contextual compatibility","Compare meaning features with current sentence context and supplied context evidence.","C_m=cosine(mu_m,context)","Context-sensitive frontal/temporal interactions bias compatible semantic features."),
("Semantic ambiguity","Select or retain competing meanings","Compute posterior meaning weights and abstain when confidence/margin are insufficient.","P(m|c)=softmax(log P(m)+beta_C*C_m); gate by max and margin","Meaning selection can reflect competition; retaining alternatives models unresolved ambiguity."),
("Sentence prediction","Construct the current sentence context","Combine external context with semantic features accumulated from preceding tokens.","c_t=normalize(c_external+gamma_c*sum_{i<t}mu_i)","Distributed discourse/sentence context biases current lexical-semantic processing."),
("Sentence prediction","Generate upcoming word predictions","Generate candidate next words from supplied transition counts and feature associations.","P(w_{t+1}|c_t,w_t) proportional to n(w_t,w_{t+1})+feature_prior(w|c_t)","Predictive language processing can pre-activate likely lexical/semantic states."),
("Sentence prediction","Calculate contextual predictability","Estimate observed token probability from the preceding predictive distribution.","surprisal(w_t)=-log(P(w_t|context_{t-1})+epsilon)","Unexpected inputs can generate larger prediction errors than predictable inputs."),
("Sentence prediction","Pre-activate expected features","Pool semantic features of predicted words weighted by predicted probabilities.","mu_pred=sum_w P(w|context)*mu_w","Top-down prediction can pre-activate expected semantic feature populations."),
("Sentence prediction","Modulate predictions by reliability","Estimate prediction reliability from evidence mass and predictive entropy.","r_pred=evidence_coverage*(1-H(P)/log K)","Precision of top-down predictions varies with contextual certainty."),
("Prediction error","Construct incoming semantic representation","Combine incoming semantic evidence with posterior-weighted lexical meaning features.","mu_in=eta*mu_observed+(1-eta)*sum_m P(m|c)*mu_m","Bottom-up feature evidence and activated meanings jointly form input representation."),
("Prediction error","Calculate semantic prediction residual","Compute signed feature-wise mismatch between incoming and expected representations.","epsilon_f=mu_in(f)-mu_pred(f)","Prediction-error accounts propose mismatch signals between predicted and incoming features."),
("Prediction error","Calculate precision-weighted error","Scale residuals by sensory precision relative to total sensory and prior precision.","epsilon_pi=pi_s/(pi_s+pi_p*r_pred+epsilon)*epsilon_f","Precision weighting is a predictive-processing hypothesis, not a unique established circuit."),
("Prediction error","Update semantic interpretation","Correct the predicted semantic state toward incoming evidence using weighted residual.","mu_post=clip(mu_pred+eta_e*epsilon_pi,0,1)","Recurrent semantic representations may be revised as bottom-up evidence arrives."),
("Prediction error","Update predictions for subsequent words","Update transient sentence state and regenerate next-word predictions; do not silently train the lexicon.","c_{t+1}=normalize(c_t+gamma*mu_post); P_{t+1}=Predict(c_{t+1})","Updated semantic context can alter subsequent anticipatory activity."),
)
OPERATION_REGISTRY = tuple(OperationSpec(f"LC-{i:03d}", *row) for i,row in enumerate(_ROWS,1))
if len(OPERATION_REGISTRY) != 40 or len({x.operation_id for x in OPERATION_REGISTRY}) != 40:
    raise RuntimeError("Registry must contain 40 unique ordered operations")

def registry_as_dicts(): return [x.to_dict() for x in OPERATION_REGISTRY]
