"""Executable 40-step lexical-context and predictive-processing pipeline."""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence
import math
import re
import unicodedata

from .biology import POPULATIONS, LexicalContextLIFSimulator
from .contracts import (G2PRule, LexicalContextConfig, LexicalContextRequest,
                        LexicalContextResult, LexicalEntry, LexicalMeaning, MorphemeEntry)
from .registry import OPERATION_REGISTRY

def _sigmoid(x: float) -> float:
    x=max(-40.0,min(40.0,float(x)))
    return 1.0/(1.0+math.exp(-x))

def _softmax(scores: Sequence[float], temperature: float=1.0) -> list[float]:
    if not scores: return []
    t=max(1e-6,temperature); z=[max(-100.0,min(100.0,float(x)/t)) for x in scores]
    m=max(z); ex=[math.exp(x-m) for x in z]; total=sum(ex)
    return [x/total for x in ex] if total else [1.0/len(z)]*len(z)

def _levenshtein(a: str, b: str) -> int:
    if len(a)<len(b): a,b=b,a
    row=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        nxt=[i]
        for j,cb in enumerate(b,1):
            nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(ca!=cb)))
        row=nxt
    return row[-1]

def _cosine(a: Mapping[str,float], b: Mapping[str,float]) -> float:
    keys=set(a)|set(b)
    if not keys: return 0.0
    dot=sum(float(a.get(k,0))*float(b.get(k,0)) for k in keys)
    na=math.sqrt(sum(float(a.get(k,0))**2 for k in keys)); nb=math.sqrt(sum(float(b.get(k,0))**2 for k in keys))
    return dot/(na*nb) if na and nb else 0.0

def _normalize_features(features: Mapping[str,float]) -> dict[str,float]:
    if not features: return {}
    peak=max(float(v) for v in features.values())
    return {str(k):max(0.0,min(1.0,float(v)/peak)) for k,v in features.items()} if peak else {}

def _blend_features(*maps: Mapping[str,float], weights: Sequence[float] | None=None) -> dict[str,float]:
    if not maps: return {}
    ws=list(weights or [1.0]*len(maps)); out: dict[str,float]={}
    for m,w in zip(maps,ws):
        for k,v in m.items(): out[k]=out.get(k,0.0)+float(w)*float(v)
    return {k:max(0.0,min(1.0,v)) for k,v in out.items()}

def _ngrams(word: str, n: int=2) -> set[str]:
    padded=f"^{word}$"
    return {padded[i:i+n] for i in range(max(0,len(padded)-n+1))}

def _dice(a: str,b: str) -> float:
    x,y=_ngrams(a),_ngrams(b)
    return 2*len(x&y)/(len(x)+len(y)) if x or y else 0.0

class LexicalContextProcessor:
    """Processes words using supplied lexicon, morphemes, G2P rules and transition evidence."""
    def __init__(self, lexicon: Sequence[LexicalEntry], *, morphemes: Sequence[MorphemeEntry]=(),
                 g2p_rules: Sequence[G2PRule]=(), bigram_counts: Mapping[str,Mapping[str,float]]|None=None,
                 config: LexicalContextConfig|None=None):
        self.entries=tuple(lexicon); self.morphemes=tuple(morphemes); self.g2p_rules=tuple(g2p_rules)
        if not self.entries: raise ValueError("lexicon must contain at least one entry")
        if len({e.entry_id for e in self.entries})!=len(self.entries): raise ValueError("entry_id values must be unique")
        self.config=config or LexicalContextConfig()
        self._by_form: dict[str,list[LexicalEntry]]=defaultdict(list)
        self._by_id={e.entry_id:e for e in self.entries}
        self._morpheme_by_id={m.morpheme_id:m for m in self.morphemes}
        for entry in self.entries: self._by_form[entry.orthography.casefold()].append(entry)
        self._bigram={str(a).casefold():{str(b).casefold():max(0.0,float(n)) for b,n in row.items()}
                      for a,row in (bigram_counts or {}).items()}
        self._exposure=Counter({e.entry_id:float(e.exposure_count) for e in self.entries})
        self._last_activation: dict[str,tuple[float,float]]={}
        self._adaptation: dict[str,float]=defaultdict(float)
        self._session_clock_ms=0.0
    @property
    def model_fingerprint(self) -> tuple:
        return tuple((e.entry_id,e.orthography,e.frequency,tuple(e.meanings)) for e in self.entries)
    def _entry_for_morpheme_stem(self, stem: str) -> list[LexicalEntry]:
        return self._by_form.get(stem.casefold(),[])
    def _morph_splits(self, word: str) -> list[dict[str,Any]]:
        found=[]
        prefixes=[m for m in self.morphemes if m.morpheme_type=="prefix"]
        suffixes=[m for m in self.morphemes if m.morpheme_type=="suffix"]
        for pre in [None,*prefixes]:
            if pre and not word.startswith(pre.form): continue
            after=word[len(pre.form):] if pre else word
            for suf in [None,*suffixes]:
                if suf and not after.endswith(suf.form): continue
                stem=after[:-len(suf.form)] if suf else after
                if not stem or stem==word: continue
                roots=self._entry_for_morpheme_stem(stem)
                if roots:
                    found.append({"prefix":pre.morpheme_id if pre else None,"stem":stem,
                                  "suffix":suf.morpheme_id if suf else None,
                                  "morpheme_ids":[x for x in (pre.morpheme_id if pre else None,
                                                               suf.morpheme_id if suf else None) if x],
                                  "root_entry_ids":[x.entry_id for x in roots],
                                  "fit":(math.log1p(pre.frequency if pre else 1.0)+
                                         math.log1p(suf.frequency if suf else 1.0)+
                                         sum(math.log1p(x.frequency) for x in roots))/3.0})
        found.sort(key=lambda x:x["fit"],reverse=True)
        return found[:self.config.max_candidates]
    def _pronunciations(self, entry: LexicalEntry|None, word: str) -> list[dict[str,Any]]:
        alternatives=[]
        if entry and entry.phonemes:
            alternatives.append({"phonemes":list(entry.phonemes),"source":"lexicon","reliability":1.0})
        # Greedy longest-grapheme rule coverage. Multiple exact paths remain explicit alternatives.
        rules=[r for r in self.g2p_rules if r.language in {"und",entry.language if entry else "und"}]
        if rules:
            memo: dict[int,list[tuple[list[str],float]]] = {len(word):[([],1.0)]}
            for i in range(len(word)-1,-1,-1):
                options=[]
                for rule in rules:
                    if word.startswith(rule.graphemes,i):
                        for tail,score in memo.get(i+len(rule.graphemes),[])[:4]:
                            options.append((list(rule.phonemes)+tail,rule.reliability*score))
                if not options: memo[i]=[]
                else:
                    options.sort(key=lambda x:x[1],reverse=True); memo[i]=options[:4]
            for phones,score in memo.get(0,[])[:4]:
                alternatives.append({"phonemes":phones,"source":"g2p_rules","reliability":score})
        unique={}
        for alt in alternatives:
            key=tuple(alt["phonemes"])
            if key not in unique or alt["reliability"]>unique[key]["reliability"]: unique[key]=alt
        return list(unique.values())
    def _meaning_rows(self, entry: LexicalEntry, context: Mapping[str,float]) -> list[dict[str,Any]]:
        meanings=list(entry.meanings)
        if not meanings: return []
        raw=[max(1e-9,m.prior) for m in meanings]
        priors=[x/sum(raw) for x in raw]
        compatibility=[max(0.0,_cosine(m.features,context)) for m in meanings]
        posterior=_softmax([math.log(p)+self.config.context_gain*c for p,c in zip(priors,compatibility)],0.25)
        rows=[]
        for meaning,prior,fit,post in zip(meanings,priors,compatibility,posterior):
            rows.append({"meaning_id":meaning.meaning_id,"prior":prior,"contextual_compatibility":fit,
                         "posterior":post,"features":dict(meaning.features),"verified_by":meaning.verified_by,
                         "source_reference":meaning.source_reference})
        rows.sort(key=lambda x:x["posterior"],reverse=True)
        return rows
    def process(self, request: LexicalContextRequest) -> LexicalContextResult:
        if not isinstance(request,LexicalContextRequest): raise TypeError("request must be LexicalContextRequest")
        original=request.text if request.text is not None else " ".join(request.tokens or ())
        normalized=unicodedata.normalize("NFKC",original).casefold()
        words=list(request.tokens) if request.tokens is not None else re.findall(r"[^\W_]+(?:['’][^\W_]+)?",normalized,flags=re.UNICODE)
        words=[unicodedata.normalize("NFKC",w).casefold() for w in words if w.strip()]
        if not words: raise ValueError("request contains no lexical tokens")
        cfg=self.config; now=request.timestamp_ms if request.timestamp_ms is not None else self._session_clock_ms
        max_count=max([e.frequency+self._exposure[e.entry_id] for e in self.entries]+[1.0])
        context=dict(request.context_features)
        predicted_next: dict[str,float]={}
        results=[]; trace=[]; neural_trace=[]
        simulator=LexicalContextLIFSimulator(cfg) if cfg.simulate_biology else None
        session_events: dict[str,list[tuple[float,float]]]=defaultdict(list)
        for token_index,word in enumerate(words):
            rows_by_step: list[dict[str,Any]]=[]
            def mark(step: int, value: Any, drives: Mapping[str,float]|None=None):
                spec=OPERATION_REGISTRY[step-1]
                row={"operation_id":spec.operation_id,"name":spec.name,"subsystem":spec.subsystem,
                     "token_index":token_index,"token":word,"output":value}
                rows_by_step.append(row); trace.append(row)
                drive_map={p:0.0 for p in POPULATIONS}
                if drives:
                    for k,v in drives.items(): drive_map[k]=max(0.0,min(1.0,float(v)))
                if simulator:
                    neural_trace.append(simulator.step(spec.operation_id,[drive_map]))
                else:
                    neural_trace.append({"operation_id":spec.operation_id,"status":"skipped"})
            # 1 Retrieve accumulated exposure; 2 logarithmic frequency; 3 resting activation.
            exact=self._by_form.get(word,[])
            ranked=[]
            for e in self.entries:
                exposure=e.frequency+self._exposure[e.entry_id]
                rank=(math.log1p(exposure)/math.log1p(max_count))
                sim=max(0.0,1.0-_levenshtein(word,e.orthography.casefold())/max(1,len(word),len(e.orthography)))
                ranked.append((e,exposure,rank,sim))
            ranked.sort(key=lambda z:z[3],reverse=True)
            exact_ids={e.entry_id for e in exact}
            ranked=[z for z in ranked if z[3]>=cfg.similarity_floor or z[0].entry_id in exact_ids][:cfg.max_candidates]
            exposures={e.entry_id:n for e,n,_,_ in ranked}
            mark(1,{"exposure_by_entry":exposures,"session_exposure":dict(self._exposure)},
                 {"lexical":0.1})
            freq_strength={e.entry_id:f for e,_,f,_ in ranked}
            mark(2,{"log_frequency_strength":freq_strength},{"lexical":sum(freq_strength.values())/max(1,len(freq_strength))})
            resting={e.entry_id:cfg.base_resting_activation+cfg.frequency_gain*f for e,_,f,_ in ranked}
            mark(3,{"resting_activation":resting},{"lexical":sum(resting.values())/max(1,len(resting))})
            # Repetition state and temporal decay.
            thresholds={}; history={}; decays={}; facilitations={}; adaptations={}
            activations={}
            for e,n,f,s in ranked:
                last=self._last_activation.get(e.entry_id)
                elapsed=max(0.0,now-last[0]) if last else float("inf")
                decay=math.exp(-elapsed/cfg.history_tau_ms) if math.isfinite(elapsed) else 0.0
                recent=[x for x in session_events[e.entry_id] if now-x[0]<=cfg.history_tau_ms]
                history[e.entry_id]=[{"time_ms":t,"activation":a} for t,a in recent[-cfg.max_recent_repetitions:]]
                adaptation=self._adaptation[e.entry_id]*(math.exp(-elapsed/cfg.adaptation_tau_ms) if math.isfinite(elapsed) else 0.0)
                threshold=cfg.threshold_base-cfg.threshold_frequency_gain*f+cfg.threshold_adaptation_gain*adaptation
                facil=cfg.repetition_gain*(math.log1p(len(recent))/math.log1p(cfg.max_recent_repetitions))*decay
                stimulus=cfg.stimulus_gain*s
                base=resting[e.entry_id]
                raw=_sigmoid((base+stimulus+facil-threshold-cfg.adaptation_gain*adaptation)/cfg.activation_temperature)
                thresholds[e.entry_id]=threshold; decays[e.entry_id]=decay
                facilitations[e.entry_id]=facil; adaptations[e.entry_id]=adaptation; activations[e.entry_id]=raw
            mark(4,{"thresholds":thresholds,"adaptation":adaptations},{"lexical":sum(adaptations.values())/max(1,len(adaptations))})
            mark(5,{"frequency_conditioned_activation":activations},{"lexical":sum(activations.values())/max(1,len(activations))})
            mark(6,{"activation_history":history},{"lexical":sum(len(x) for x in history.values())/max(1,len(history))})
            similarity={e.entry_id:s for e,_,_,s in ranked}
            mark(7,{"stimulus_similarity":similarity},{"orthographic":max(similarity.values(),default=0.0)})
            mark(8,{"temporal_decay":decays},{"lexical":sum(decays.values())/max(1,len(decays))})
            mark(9,{"repetition_facilitation":facilitations},{"lexical":sum(facilitations.values())/max(1,len(facilitations))})
            mark(10,{"neural_adaptation":adaptations},{"lexical":sum(adaptations.values())/max(1,len(adaptations))})
            # Morphological boundaries and composition.
            splits=self._morph_splits(word)
            mark(11,{"boundary_candidates":splits},{"morphological":0.4 if splits else 0.0})
            morph_scores=[]; morph_features=[]
            for split in splits:
                roots=[self._by_id[x] for x in split["root_entry_ids"]]
                root_score=max((activations.get(r.entry_id,0.35) for r in roots),default=0.0)
                affixes=[self._morpheme_by_id[str(x)] for x in split["morpheme_ids"]]
                affix_support=math.prod(max(0.01,a.productivity*a.frequency/(1+a.frequency)) for a in affixes) if affixes else 1.0
                morph_scores.append(root_score*affix_support*min(1.0,split["fit"]))
                feature_maps=[m.features for r in roots for m in r.meanings]+[a.semantic_features for a in affixes]
                morph_features.append(_blend_features(*feature_maps) if feature_maps else {})
            mark(12,{"morphology_scores":morph_scores},{"morphological":max(morph_scores,default=0.0)})
            morpheme_meanings=[]
            for split in splits:
                morpheme_meanings.append({"split":split,"features":morph_features[splits.index(split)]})
            mark(13,{"morpheme_meanings":morpheme_meanings},{"semantic":0.2 if morpheme_meanings else 0.0})
            composition=[{"split":s,"features":f,"score":sc} for s,f,sc in zip(splits,morph_features,morph_scores)]
            mark(14,{"composed_morphology":composition},{"semantic":max(morph_scores,default=0.0)})
            # Candidate combination gives exact word forms and their morphological root forms.
            candidate_scores={}
            for e,_,_,_ in ranked: candidate_scores[e.entry_id]=activations[e.entry_id]*cfg.whole_word_gain
            for split,score in zip(splits,morph_scores):
                for root_id in split["root_entry_ids"]:
                    candidate_scores[root_id]=max(candidate_scores.get(root_id,0.0),score*cfg.morphology_gain)
            total_routes=sum(candidate_scores.values()) or 1.0
            blended={k:v/total_routes for k,v in candidate_scores.items()}
            mark(15,{"whole_word_morphology_integration":blended},{"lexical":max(blended.values(),default=0.0),"morphological":max(morph_scores,default=0.0)})
            # G2P and mapping alternatives.
            pronunciation_rows=[]
            for e,_,_,_ in ranked:
                alternatives=self._pronunciations(e,word if e.orthography.casefold()==word else e.orthography.casefold())
                if alternatives:
                    max_rel=max(a["reliability"] for a in alternatives)
                    for a in alternatives:
                        pronunciation_rows.append({"entry_id":e.entry_id,**a,"mapping_consistency":a["reliability"]})
            mark(16,{"grapheme_phoneme_correspondences":pronunciation_rows},
                 {"phonological":min(1.0,len(pronunciation_rows)/3)})
            consistency={x["entry_id"]:x["mapping_consistency"] for x in pronunciation_rows}
            mark(17,{"mapping_consistency":consistency},{"phonological":sum(consistency.values())/max(1,len(consistency))})
            by_entry=defaultdict(list)
            for x in pronunciation_rows: by_entry[x["entry_id"]].append(x)
            mark(18,{"competing_pronunciations":dict(by_entry)},{"phonological":0.5 if pronunciation_rows else 0.0})
            pronunciation_weights={}
            for entry_id,alts in by_entry.items():
                entry=self._by_id[entry_id]
                weights=_softmax([math.log(max(1e-9,entry.frequency))+math.log(max(1e-9,a["reliability"])) for a in alts],0.2)
                pronunciation_weights[entry_id]=[{**a,"weight":w} for a,w in zip(alts,weights)]
            mark(19,{"pronunciation_alternatives":pronunciation_weights},{"phonological":max((a["reliability"] for a in pronunciation_rows),default=0.0)})
            phonological_fit={}
            for e,_,_,_ in ranked:
                alts=pronunciation_weights.get(e.entry_id,[])
                phonological_fit[e.entry_id]=max((a["reliability"] for a in alts),default=0.0)
            mark(20,{"phonological_lexical_feedback":phonological_fit},
                 {"phonological":max(phonological_fit.values(),default=0.0),"lexical":sum(activations.values())/max(1,len(activations))})
            # Orthographic neighborhood and competition.
            neighbors={}
            for e,_,_,_ in ranked:
                neighbors[e.entry_id]=[x.entry_id for x in self.entries if x.entry_id!=e.entry_id
                    and _levenshtein(e.orthography.casefold(),x.orthography.casefold())<=cfg.max_edit_distance]
            mark(21,{"orthographic_neighbors":neighbors},{"orthographic":0.3 if neighbors else 0.0})
            neighborhood={}
            for e,_,_,_ in ranked:
                ns=neighbors[e.entry_id]
                neighborhood[e.entry_id]=sum((math.log1p(self._by_id[x].frequency)*activations.get(x,0.05)) for x in ns)/(len(ns)+1e-6)
            mark(22,{"neighborhood_strength":neighborhood},{"orthographic":min(1.0,sum(neighborhood.values())/max(1,len(neighborhood)))})
            shared={}
            for e,_,_,_ in ranked:
                shared[e.entry_id]=sum(_dice(e.orthography.casefold(),self._by_id[x].orthography.casefold())*activations.get(x,0.0)
                                       for x in neighbors[e.entry_id])
            mark(23,{"shared_orthographic_activation":shared},{"orthographic":min(1.0,sum(shared.values())/max(1,len(shared)))})
            inhibited={e.entry_id:max(0.0,activations[e.entry_id]-cfg.lateral_inhibition*shared[e.entry_id]) for e,_,_,_ in ranked}
            mark(24,{"competitive_inhibition":inhibited},{"lexical":max(inhibited.values(),default=0.0)})
            settling=dict(inhibited)
            for _ in range(cfg.settle_steps):
                nxt={}
                for e,_,_,_ in ranked:
                    excit=cfg.recurrent_excitation*settling.get(e.entry_id,0.0)
                    inhib=cfg.lateral_inhibition*sum(settling.get(x,0.0) for x in neighbors[e.entry_id])
                    target=_sigmoid(4*(inhibited[e.entry_id]+excit-inhib)-2)
                    nxt[e.entry_id]=(1-cfg.settle_damping)*settling[e.entry_id]+cfg.settle_damping*target
                settling=nxt
            lexical_order=sorted(ranked,key=lambda z:settling.get(z[0].entry_id,0.0),reverse=True)
            mark(25,{"settled_activation":settling,"ranked_entry_ids":[x[0].entry_id for x in lexical_order]},
                 {"lexical":max(settling.values(),default=0.0),"orthographic":max(similarity.values(),default=0.0)})
            # Select the most-supported whole-word candidate but preserve alternatives.
            candidate_records=[]
            for e,_,_,s in lexical_order:
                candidate_records.append({"entry_id":e.entry_id,"orthography":e.orthography,"activation":settling[e.entry_id],
                    "similarity":s,"frequency":e.frequency,"lexical_category":e.lexical_category,
                    "phonological_fit":phonological_fit.get(e.entry_id,0.0),
                    "neighborhood_strength":neighborhood.get(e.entry_id,0.0)})
            selected_entry=self._by_id[candidate_records[0]["entry_id"]] if candidate_records else None
            # 26-30 meaning ambiguity and selection.
            meaning_candidates=self._meaning_rows(selected_entry,context) if selected_entry else []
            ambiguous=len(meaning_candidates)>1
            mark(26,{"is_ambiguous":ambiguous,"meaning_count":len(meaning_candidates)},
                 {"semantic":0.4 if ambiguous else 0.1})
            mark(27,{"meaning_priors":[{"meaning_id":m["meaning_id"],"prior":m["prior"]} for m in meaning_candidates]},
                 {"semantic":max((m["prior"] for m in meaning_candidates),default=0.0)})
            mark(28,{"meaning_features":{m["meaning_id"]:m["features"] for m in meaning_candidates}},
                 {"semantic":min(1.0,len(meaning_candidates)/3)})
            mark(29,{"contextual_compatibility":{m["meaning_id"]:m["contextual_compatibility"] for m in meaning_candidates}},
                 {"context_predictive":max((m["contextual_compatibility"] for m in meaning_candidates),default=0.0)})
            ranked_meanings=meaning_candidates
            best=ranked_meanings[0] if ranked_meanings else None
            margin=(best["posterior"]-ranked_meanings[1]["posterior"]) if len(ranked_meanings)>1 else (best["posterior"] if best else 0.0)
            abstained=bool(best and (best["posterior"]<cfg.confidence_threshold or (len(ranked_meanings)>1 and margin<cfg.ambiguity_margin)))
            chosen=None if abstained or not best else best
            selected_features=dict(chosen["features"]) if chosen else {}
            mark(30,{"selected_meaning":chosen["meaning_id"] if chosen else None,"abstained":abstained,
                     "posterior_margin":margin,"retained_alternatives":ranked_meanings},
                 {"semantic":best["posterior"] if best else 0.0})
            # 31-35 sentence context and anticipatory representation.
            prior_context=dict(context)
            mark(31,{"context_before_token":prior_context},{"context_predictive":min(1.0,len(context)/5)})
            prev=words[token_index-1] if token_index else ""
            transition=self._bigram.get(prev,{}) if prev else {}
            predicted_counts={k:v for k,v in transition.items() if v>0}
            if not predicted_counts:
                predicted_counts={e.orthography.casefold():max(0.0,e.frequency) for e in self.entries
                                  if _cosine({f:1.0 for f in context},_blend_features(*(m.features for m in e.meanings)))>0}
            if not predicted_counts: predicted_counts={e.orthography.casefold():e.frequency for e in self.entries[:min(5,len(self.entries))]}
            total_pred=sum(predicted_counts.values()) or 1.0
            predicted={k:v/total_pred for k,v in predicted_counts.items()}
            mark(32,{"upcoming_word_predictions":predicted},{"context_predictive":max(predicted.values(),default=0.0)})
            observed_probability=predicted.get(word,0.0)
            if token_index==0 and not observed_probability: observed_probability=1.0/max(1,len(predicted))
            surprisal=-math.log(max(1e-9,observed_probability))
            mark(33,{"contextual_predictability":observed_probability,"surprisal_nats":surprisal},
                 {"context_predictive":max(0.0,1.0-min(1.0,surprisal/10.0))})
            predicted_features={}
            for pred_word,p in predicted.items():
                for e in self._by_form.get(pred_word,[]):
                    feats=_blend_features(*(m.features for m in e.meanings))
                    for f,v in feats.items(): predicted_features[f]=predicted_features.get(f,0.0)+p*v
            predicted_features={k:min(1.0,v) for k,v in predicted_features.items()}
            mark(34,{"preactivated_expected_features":predicted_features},{"context_predictive":min(1.0,sum(predicted_features.values()))})
            evidence_coverage=min(1.0,sum(predicted.values()))
            entropy=-sum(p*math.log(max(p,1e-9)) for p in predicted.values())
            max_entropy=math.log(max(1,len(predicted)))
            reliability=evidence_coverage*(1.0-entropy/max_entropy) if max_entropy else 1.0
            mark(35,{"prediction_reliability":reliability,"predictive_entropy_nats":entropy},
                 {"context_predictive":reliability})
            # 36-40 semantic input, error, precision and context update.
            incoming=dict(request.incoming_semantic_evidence.get(token_index,selected_features))
            if not incoming: incoming=dict(selected_features)
            incoming_features=_blend_features(incoming,selected_features,weights=(0.7,0.3))
            mark(36,{"incoming_semantic_representation":incoming_features},{"semantic":min(1.0,sum(incoming_features.values()))})
            feature_keys=set(incoming_features)|set(predicted_features)
            residual={f:incoming_features.get(f,0.0)-predicted_features.get(f,0.0) for f in feature_keys}
            mark(37,{"semantic_prediction_residual":residual},{"semantic":min(1.0,sum(abs(v) for v in residual.values()))})
            precision_factor=cfg.sensory_precision/(cfg.sensory_precision+cfg.prior_precision*reliability+1e-9)
            weighted={f:precision_factor*v for f,v in residual.items()}
            mark(38,{"precision_factor":precision_factor,"precision_weighted_error":weighted},
                 {"semantic":min(1.0,sum(abs(v) for v in weighted.values()))})
            post={f:max(0.0,min(1.0,predicted_features.get(f,0.0)+cfg.prediction_error_gain*v))
                  for f,v in weighted.items()}
            mark(39,{"updated_semantic_interpretation":post},{"semantic":min(1.0,sum(post.values()))})
            context=_blend_features(context,post,weights=(1.0,0.7))
            context=_normalize_features(context)
            predicted_next=dict(self._bigram.get(word,{}))
            if predicted_next:
                z=sum(predicted_next.values()) or 1.0; predicted_next={k:v/z for k,v in predicted_next.items()}
            mark(40,{"updated_sentence_context":context,"next_word_predictions":predicted_next},
                 {"context_predictive":min(1.0,sum(context.values()))})
            # Store the observed word-form state as an explicit, optionally writable exposure trace.
            if selected_entry:
                session_events[selected_entry.entry_id].append((now,settling.get(selected_entry.entry_id,0.0)))
                self._last_activation[selected_entry.entry_id]=(now,settling.get(selected_entry.entry_id,0.0))
                self._adaptation[selected_entry.entry_id]=(self._adaptation[selected_entry.entry_id]*
                    math.exp(-cfg.token_interval_ms/cfg.adaptation_tau_ms)+cfg.adaptation_gain*settling.get(selected_entry.entry_id,0.0))
                if cfg.record_exposures: self._exposure[selected_entry.entry_id]+=1.0
            results.append({"token_index":token_index,"orthography":word,"candidate_words":candidate_records,
                "selected_entry":({"entry_id":selected_entry.entry_id,"orthography":selected_entry.orthography} if selected_entry else None),
                "pronunciation_alternatives":pronunciation_weights,"morphological_candidates":composition,
                "meaning_candidates":ranked_meanings,"selected_meaning":chosen,"abstained":abstained,
                "sentence_context_before":prior_context,"predicted_next_words":predicted,
                "predictability":observed_probability,"surprisal_nats":surprisal,"prediction_reliability":reliability,
                "predicted_features":predicted_features,"incoming_features":incoming_features,
                "prediction_residual":residual,"precision_weighted_error":weighted,"updated_semantics":post,
                "operation_count":len(rows_by_step)})
            now+=cfg.token_interval_ms
        self._session_clock_ms=max(self._session_clock_ms,now)
        if simulator:
            biology=simulator.snapshot(); biology["operation_trace"]=neural_trace
        else:
            biology={"enabled":False,"operation_trace":[{"operation_id":x["operation_id"],"status":"skipped"} for x in neural_trace],
                     "simulated_time_ms":0.0,"numerically_valid":True,"layers":{}}
        diagnostics={"valid":len(trace)==40*len(words),"operation_count":len(trace),
                     "operations_per_token":40,"registry_operation_count":len(OPERATION_REGISTRY),
                     "token_count":len(words),"neural_operation_count":len(neural_trace),
                     "biological_numerically_valid":biology.get("numerically_valid",True)}
        return LexicalContextResult(request.sequence_id,original,normalized,results,context,biology,trace,diagnostics,
            {"language":request.language,"source_reference":request.source_reference,
             "parameters":cfg.to_dict(),"exposure_state":"updated" if cfg.record_exposures else "read_only",
             "biological_scope":"generic LIF reference; not a fitted human-brain model"})
