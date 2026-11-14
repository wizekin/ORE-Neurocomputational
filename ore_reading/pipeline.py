"""Functional Integrated Reading Algorithm with an optional LIF biology branch."""
from __future__ import annotations
from collections import Counter, defaultdict
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence
import math
import unicodedata
import numpy as np
from .biology import BiologicalReadingSimulator, LIFConfig
from .contracts import G2PRule, ReadingConfig, ReadingLexiconEntry, ReadingRequest, ReadingResult
from .registry import OPERATION_REGISTRY

_EPS = 1e-9
_VOWEL_PHONEMES = {"AA","AE","AH","AO","AW","AY","EH","ER","EY","IH","IY","OW","OY","UH","UW","a","e","i","o","u"}

def default_g2p_rules() -> tuple[G2PRule, ...]:
    pairs = [
        ("tch",("CH",),.95),("igh",("AY",),.92),("tion",("SH","AH","N"),.90),
        ("sh",("SH",),.96),("ch",("CH",),.95),("th",("TH",),.92),("ph",("F",),.95),
        ("ng",("NG",),.92),("ck",("K",),.94),("qu",("K","W"),.90),("ee",("IY",),.88),
        ("ea",("IY",),.72),("oo",("UW",),.78),("ai",("EY",),.82),("ay",("EY",),.86),
        ("oa",("OW",),.84),("ow",("AW",),.68),("ou",("AW",),.65),("oi",("OY",),.88),
        ("oy",("OY",),.88),("ar",("AA","R"),.72),("er",("ER",),.72),("ir",("ER",),.70),
        ("ur",("ER",),.70),("a",("AE",),.55),("b",("B",),.96),("c",("K",),.78),
        ("d",("D",),.96),("e",("EH",),.55),("f",("F",),.96),("g",("G",),.78),
        ("h",("HH",),.93),("i",("IH",),.55),("j",("JH",),.96),("k",("K",),.96),
        ("l",("L",),.96),("m",("M",),.96),("n",("N",),.96),("o",("AA",),.55),
        ("p",("P",),.96),("r",("R",),.96),("s",("S",),.96),("t",("T",),.96),
        ("u",("AH",),.55),("v",("V",),.96),("w",("W",),.93),("x",("K","S"),.88),
        ("y",("Y",),.75),("z",("Z",),.96),
    ]
    return tuple(G2PRule(g,p,c,"ore-reading-default-en-v1") for g,p,c in pairs)

def _softmax(values: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    vector = np.asarray(values,dtype=np.float64)
    if vector.size == 0: return vector
    scaled = vector / max(float(temperature),_EPS)
    scaled -= np.max(scaled)
    exp = np.exp(np.clip(scaled,-60.0,60.0))
    total = float(exp.sum())
    return exp/total if total > _EPS and math.isfinite(total) else np.full(vector.size,1.0/vector.size)

def _unicode_graphemes(text: str) -> tuple[str,...]:
    clusters = []
    for char in text:
        if clusters and (unicodedata.combining(char) or char == "\ufe0f"): clusters[-1] += char
        else: clusters.append(char)
    return tuple(clusters)

def _split_punctuation(token: str) -> tuple[str,str,str]:
    start,end=0,len(token)
    while start<end and unicodedata.category(token[start]).startswith("P"): start+=1
    while end>start and unicodedata.category(token[end-1]).startswith("P"): end-=1
    return token[:start],token[start:end],token[end:]

def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left)<len(right): left,right=right,left
    previous=list(range(len(right)+1))
    for i,a in enumerate(left,1):
        current=[i]
        for j,b in enumerate(right,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]

def _similarity(left: Sequence[str],right: Sequence[str])->float:
    return max(0.0,1.0-_levenshtein(left,right)/max(len(left),len(right),1))

def _cosine(left: Mapping[str,float],right: Mapping[str,float])->float:
    keys=set(left)|set(right)
    if not keys:return 0.0
    a=np.asarray([float(left.get(k,0.0)) for k in keys])
    b=np.asarray([float(right.get(k,0.0)) for k in keys])
    denom=float(np.linalg.norm(a)*np.linalg.norm(b))
    return float(np.dot(a,b)/denom) if denom>_EPS else 0.0

def _normalize_map(values: Mapping[str,float])->dict[str,float]:
    total=sum(max(0.0,float(v)) for v in values.values())
    return {str(k):max(0.0,float(v))/total for k,v in values.items() if float(v)>0} if total>_EPS else {}

def _phoneme_features(phoneme: str)->dict[str,float]:
    p=phoneme.upper()
    vowels={"AA":(1,0,0),"AE":(.8,.1,0),"AH":(.5,0,0),"AO":(.3,0,1),"AW":(.4,.2,.7),
            "AY":(.8,.2,.1),"EH":(.6,.4,0),"ER":(.5,.1,.6),"EY":(.7,.5,0),"IH":(.8,.7,0),
            "IY":(1,1,0),"OW":(.5,.5,1),"OY":(.4,.4,.8),"UH":(.3,.7,1),"UW":(.2,1,1)}
    if p in vowels:
        h,f,r=vowels[p]; return {"vowel":1.0,"height":h,"frontness":f,"roundness":r}
    if p in {"P","B","M"}: place="bilabial"
    elif p in {"F","V","TH","DH"}: place="labiodental_or_dental"
    elif p in {"T","D","S","Z","N","L"}: place="alveolar"
    elif p in {"SH","ZH","CH","JH"}: place="postalveolar"
    elif p in {"K","G","NG"}: place="velar"
    elif p=="R": place="rhotic"
    elif p=="HH": place="glottal"
    else: place="unknown"
    voiced=p in {"B","D","G","V","DH","Z","ZH","JH","M","N","NG","L","R","W","Y"}
    fricative=p in {"F","V","TH","DH","S","Z","SH","ZH","HH"}
    stop=p in {"P","B","T","D","K","G","CH","JH"}
    return {"consonant":1.0,"voiced":float(voiced),"fricative":float(fricative),
            "stop_or_affricate":float(stop),f"place:{place}":1.0}

class IntegratedReadingProcessor:
    """Run the 72-operation functional reading registry and optional LIF branch."""
    def __init__(self,lexicon: Iterable[ReadingLexiconEntry]=(),g2p_rules: Iterable[G2PRule]|None=None,
                 config: ReadingConfig|None=None)->None:
        self.config=config or ReadingConfig()
        self.lexicon=tuple(lexicon)
        if len({e.entry_id for e in self.lexicon})!=len(self.lexicon): raise ValueError("lexicon entry_id values must be unique")
        self._lexicon_by_form=defaultdict(list)
        self._lexicon_by_first_grapheme=defaultdict(list)
        for entry in self.lexicon:
            norm=unicodedata.normalize("NFKC",entry.orthography).casefold()
            self._lexicon_by_form[norm].append(entry)
            gs=_unicode_graphemes(norm)
            if gs:self._lexicon_by_first_grapheme[gs[0]].append(entry)
        self.g2p_rules=tuple(g2p_rules) if g2p_rules is not None else default_g2p_rules()
        if len({(r.graphemes.casefold(),r.phonemes) for r in self.g2p_rules})!=len(self.g2p_rules):
            raise ValueError("G2P rules must be unique by grapheme/phoneme mapping")
        self._sorted_rules=tuple(sorted(self.g2p_rules,key=lambda r:(-len(r.graphemes),r.graphemes)))
        self._alphabet=tuple(sorted({g for e in self.lexicon for g in _unicode_graphemes(e.orthography.casefold())}))
        self._phoneme_inventory=tuple(sorted({p for e in self.lexicon for p in e.phonemes}))
        self._semantic_inventory=tuple(sorted({k for e in self.lexicon for k in e.semantic_features}))
        self._w_op,self._w_os,self._w_ps=self._build_triangle_weights()

    def _build_triangle_weights(self):
        no,np_,ns=len(self._alphabet),len(self._phoneme_inventory),len(self._semantic_inventory)
        wop=np.zeros((no,np_)); wos=np.zeros((no,ns)); wps=np.zeros((np_,ns))
        if not self.lexicon:return wop,wos,wps
        oi={v:i for i,v in enumerate(self._alphabet)}; pi={v:i for i,v in enumerate(self._phoneme_inventory)}
        si={v:i for i,v in enumerate(self._semantic_inventory)}
        for e in self.lexicon:
            weight=math.log1p(e.frequency)
            for g in set(_unicode_graphemes(e.orthography.casefold())):
                for p in set(e.phonemes):wop[oi[g],pi[p]]+=weight
                for f,v in e.semantic_features.items():wos[oi[g],si[f]]+=weight*float(v)
            for p in set(e.phonemes):
                for f,v in e.semantic_features.items():wps[pi[p],si[f]]+=weight*float(v)
        for matrix in (wop,wos,wps):
            sums=matrix.sum(axis=1,keepdims=True); np.divide(matrix,sums,out=matrix,where=sums>_EPS)
        return wop,wos,wps

    def _g2p(self,graphemes):
        joined="".join(graphemes).casefold(); phones=[]; chunks=[]; confidences=[]; i=0
        while i<len(joined):
            rule=next((r for r in self._sorted_rules if joined.startswith(r.graphemes.casefold(),i)),None)
            if rule is None:
                phones.append("<UNK_PHONEME>"); chunks.append({"graphemes":joined[i],"phonemes":["<UNK_PHONEME>"],
                    "confidence":0.0,"source_reference":"unmapped"}); confidences.append(0.0); i+=1; continue
            phones.extend(rule.phonemes); chunks.append({"graphemes":rule.graphemes,"phonemes":list(rule.phonemes),
                "confidence":float(rule.confidence),"source_reference":rule.source_reference})
            confidences.append(float(rule.confidence)); i+=len(rule.graphemes)
        coverage=sum(x["confidence"]>0 for x in chunks)/max(len(chunks),1)
        return tuple(phones),chunks,float(coverage*(np.mean(confidences) if confidences else 0.0))

    def _entry_candidates(self,word,graphemes):
        found=[]
        for e in self.lexicon:
            sim=_similarity(graphemes,_unicode_graphemes(e.orthography.casefold()))
            exact=word==unicodedata.normalize("NFKC",e.orthography).casefold()
            if exact or sim>=.5:found.append({"entry":e,"similarity":float(sim),"exact":exact,"frequency":float(e.frequency)})
        found.sort(key=lambda x:(x["exact"],x["similarity"],math.log1p(x["frequency"]),x["entry"].entry_id),reverse=True)
        return found[:self.config.max_candidates]

    def _interactive_settle(self,candidates):
        if not candidates:return np.zeros(0)
        sim=np.asarray([x["similarity"] for x in candidates]); freq=np.asarray([math.log1p(x["frequency"]) for x in candidates])
        base=2.2*sim+.12*freq; activation=_softmax(base)
        for _ in range(self.config.interactive_steps):
            scores=base+self.config.interactive_feedback_gain*activation-self.config.lateral_inhibition*(activation.sum()-activation)
            proposal=_softmax(scores); activation=.65*activation+.35*proposal
            activation/=max(float(activation.sum()),_EPS)
        return activation

    def _context_score(self,entry,context):return _cosine(entry.semantic_features,context)

    def _phonological_candidates(self,phones):
        found=[]
        for e in self.lexicon:
            sim=_similarity(phones,e.phonemes)
            if sim>=.35:found.append({"entry":e,"similarity":float(sim)})
        found.sort(key=lambda x:(x["similarity"],math.log1p(x["entry"].frequency)),reverse=True)
        return found[:self.config.max_candidates]

    def _bayesian_shortlist(self,phones,candidates,context):
        if not candidates:return [],np.zeros(0)
        noise=self.config.phonological_noise; logs=[]; rows=[]
        total=sum(x["entry"].frequency for x in candidates)+_EPS
        for item in candidates:
            e=item["entry"]; distance=_levenshtein(phones,e.phonemes); maxlen=max(len(phones),len(e.phonemes),1)
            likelihood=math.exp(-(distance/maxlen)*(1+3*(1-noise)))
            prior=(e.frequency+.5)/(total+.5*len(candidates)); fit=self._context_score(e,context)
            score=math.log(likelihood+_EPS)+math.log(prior+_EPS)+self.config.context_gain*fit
            logs.append(score); rows.append({"entry":e,"phonological_similarity":float(item["similarity"]),
                "edit_distance":distance,"likelihood":likelihood,"prior":prior,"context_fit":fit,"log_score":score})
        posterior=_softmax(logs,self.config.bayesian_temperature)
        for row,p in zip(rows,posterior):row["posterior"]=float(p)
        rows.sort(key=lambda x:x["posterior"],reverse=True)
        return rows,np.asarray([r["posterior"] for r in rows])

    def _triangle_for_token(self,graphemes,phonemes):
        if not self._alphabet or not self._semantic_inventory:
            return {},{}, {},{"available":False,"reason":"lexicon_has_no_semantic_feature_space"}
        oi={v:i for i,v in enumerate(self._alphabet)}; pi={v:i for i,v in enumerate(self._phoneme_inventory)}
        si={v:i for i,v in enumerate(self._semantic_inventory)}
        o=np.zeros(len(oi)); p=np.zeros(len(pi))
        for g in graphemes:
            if g in oi:o[oi[g]]+=1/max(len(graphemes),1)
        for ph in phonemes:
            if ph in pi:p[pi[ph]]+=1/max(len(phonemes),1)
        p_o=o@self._w_op; s_o=o@self._w_os; s_p=p@self._w_ps
        pstate=p.copy(); sstate=np.zeros(len(si))
        for _ in range(5):
            pp=.5*p+.5*(o@self._w_op); sp=.45*(o@self._w_os)+.45*(pstate@self._w_ps)+.10*sstate
            pstate=np.clip(.55*pstate+.45*pp,0,1); sstate=np.clip(.55*sstate+.45*sp,0,1)
        semantic={f:float(sstate[i]) for f,i in si.items() if sstate[i]>0}
        info={"available":True,"orthography_to_phonology_mass":float(p_o.sum()),
              "orthography_to_semantics_mass":float(s_o.sum()),"phonology_to_semantics_mass":float(s_p.sum()),
              "semantic_dimensions":len(si)}
        return ({f:float(s_o[i]) for f,i in si.items()},{f:float(s_p[i]) for f,i in si.items()},semantic,info)

    def process(self,request: ReadingRequest)->ReadingResult:
        started=perf_counter(); trace=[]; cursor=0; work=[]; warnings=[]; context_state=dict(request.context)
        biological={"status":"NOT_CONFIGURED"}
        if not isinstance(request,ReadingRequest):raise TypeError("process expects a ReadingRequest")
        original=request.text if request.text is not None else " ".join(request.tokens or ())
        normalized=""
        def mark(number,**details):
            nonlocal cursor
            expected=f"IR-{number:03d}"
            if cursor>=72 or OPERATION_REGISTRY[cursor].operation_id!=expected:raise RuntimeError(f"operation order violation at {expected}")
            spec=OPERATION_REGISTRY[cursor]
            trace.append({"index":cursor+1,"operation_id":spec.operation_id,"subsystem":spec.subsystem,
                "operation":spec.name,"status":"EXECUTED","details":details}); cursor+=1

        mark(1,sequence_id=request.sequence_id,input_kind="text" if request.text is not None else "tokens")
        normalized=unicodedata.normalize("NFKC",original).casefold()
        mark(2,unicode_form="NFKC",casefold_applied=True,normalized_characters=len(normalized))
        raw=list(request.tokens) if request.tokens is not None else normalized.split()
        mark(3,token_count=len(raw),order=request.reading_direction)
        for i,surface in enumerate(raw):
            prefix,core,suffix=_split_punctuation(unicodedata.normalize("NFKC",str(surface)).casefold())
            if not core:warnings.append(f"token_{i}:punctuation_only"); core=surface
            gs=_unicode_graphemes(core)
            work.append({"token_index":i,"surface":str(surface),"prefix_punctuation":prefix,"suffix_punctuation":suffix,
                         "word":core,"graphemes":gs})
        mark(4,punctuation_tokens=sum(bool(t["prefix_punctuation"] or t["suffix_punctuation"]) for t in work))
        mark(5,grapheme_counts=[len(t["graphemes"]) for t in work])
        for t in work:t["visual_letter_evidence"]={g:n/max(len(t["graphemes"]),1) for g,n in Counter(t["graphemes"]).items()}
        mark(6,evidence_mode="symbolic_orthography",confidence="not_provided")
        profile={"language":request.language,"reading_direction":request.reading_direction,
            "unicode_scripts":sorted({unicodedata.name(c,"UNKNOWN").split(" ")[0] for c in normalized if c.isalpha()})}
        mark(7,script_profile=profile)
        for t in work:
            length=max(len(t["graphemes"]),1)
            t["orthographic_codes"]=[{"grapheme":g,"position":i,"normalized_position":i/max(length-1,1)} for i,g in enumerate(t["graphemes"])]
        mark(8,position_code="normalized_serial_position",token_count=len(work))

        mark(9,entry_count=len(self.lexicon),provenance_fields="entry_id/verified_by/source_reference")
        mark(10,exact_index_size=len(self._lexicon_by_form),first_grapheme_index_size=len(self._lexicon_by_first_grapheme))
        for t in work:t["lexical_candidates"]=self._entry_candidates(t["word"],t["graphemes"])
        mark(11,exact_match_count=sum(any(x["exact"] for x in t["lexical_candidates"]) for t in work))
        mark(12,approximate_candidate_counts=[len(t["lexical_candidates"]) for t in work])
        mark(13,letter_support="grapheme_edit_similarity_and_position_overlap")
        mark(14,frequency_prior="log1p(entry.frequency)")
        for t in work:
            acts=self._interactive_settle(t["lexical_candidates"]); t["interactive_activations"]=acts
            for item,a in zip(t["lexical_candidates"],acts):item["activation"]=float(a)
        mark(15,activation_vectors=[len(t["interactive_activations"]) for t in work])
        mark(16,recurrent_steps=self.config.interactive_steps,feedback_gain=self.config.interactive_feedback_gain,lateral_inhibition=self.config.lateral_inhibition)

        mark(17,lexical_route="lookup_pronunciations_from_active_orthographic_candidates")
        mark(18,g2p_rule_count=len(self.g2p_rules),rule_order="longest_grapheme_match")
        for t in work:
            phones,chunks,conf=self._g2p(t["graphemes"]); t["sublexical_pronunciation"]=phones
            t["g2p_chunks"]=chunks; t["g2p_confidence"]=conf
        mark(19,chunks_per_token=[len(t["g2p_chunks"]) for t in work])
        mark(20,sublexical_phone_counts=[len(t["sublexical_pronunciation"]) for t in work])
        for t in work:
            best=t["lexical_candidates"][0] if t["lexical_candidates"] else None
            lexical=(float(.75 if best["exact"] else 0.0)+.25*float(best["activation"]))*float(best["similarity"]) if best else 0.0
            t["route_scores"]={"lexical":lexical,"sublexical":float(t["g2p_confidence"])}
        mark(21,route_scoring="lexical_activation_x_form_similarity_vs_g2p_coverage_confidence")
        for t in work:
            weights=_softmax([t["route_scores"]["lexical"]*self.config.lexical_route_gain,
                              t["route_scores"]["sublexical"]*self.config.sublexical_route_gain])
            t["route_weights"]={"lexical":float(weights[0]),"sublexical":float(weights[1])}
        mark(22,route_gains={"lexical":self.config.lexical_route_gain,"sublexical":self.config.sublexical_route_gain})
        for t in work:
            best=t["lexical_candidates"][0] if t["lexical_candidates"] else None
            lexical_phones=tuple(best["entry"].phonemes) if best else tuple()
            if lexical_phones and t["route_weights"]["lexical"]>=t["route_weights"]["sublexical"]:
                selected,source=lexical_phones,"lexical"
            else:selected,source=tuple(t["sublexical_pronunciation"]),"sublexical"
            t["lexical_pronunciation"]=lexical_phones;t["integrated_pronunciation"]=selected;t["pronunciation_source"]=source
        mark(23,selected_routes=[t["pronunciation_source"] for t in work])
        for t in work:
            weights=list(t["route_weights"].values()); entropy=-sum(p*math.log(p+_EPS) for p in weights)/math.log(2)
            t["pronunciation_confidence"]=float(max(weights)*(1-.25*entropy))
        mark(24,mean_pronunciation_confidence=float(np.mean([t["pronunciation_confidence"] for t in work])))

        for t in work:
            phones=t["integrated_pronunciation"]; t["phoneme_features"]=[_phoneme_features(p) for p in phones]
            t["syllable_nuclei"]=[i for i,p in enumerate(phones) if p.upper() in _VOWEL_PHONEMES]
            t["stress_pattern"]=next((list(x["entry"].stress or ()) for x in t["lexical_candidates"] if x["entry"].stress is not None),None)
            t["phonological_buffer"]=list(phones)
            t["phoneme_positions"]=[{"phoneme":p,"index":i,"normalized_position":i/max(len(phones)-1,1)} for i,p in enumerate(phones)]
            rehearsal=np.ones(len(phones))
            for _ in range(3):rehearsal=.8*rehearsal+.2*np.ones(len(phones))
            t["rehearsal_state"]=rehearsal.tolist()
            t["phonological_similarity"]=max((_similarity(phones,e.phonemes) for e in self.lexicon),default=0.0)
        mark(25,feature_dimensions="place/manner/voicing_or_vowel_features")
        mark(26,syllable_nuclei_counts=[len(t["syllable_nuclei"]) for t in work])
        mark(27,stress_available=sum(t["stress_pattern"] is not None for t in work))
        mark(28,buffer_lengths=[len(t["phonological_buffer"]) for t in work])
        mark(29,serial_position="zero_based_and_normalized")
        mark(30,rehearsal_cycles=3,retention_factor=.8)
        mark(31,similarity_metric="normalized_phoneme_edit_similarity")
        mark(32,prepared_sequences=[list(t["integrated_pronunciation"]) for t in work])

        for t in work:
            phones=t["integrated_pronunciation"]
            t["trace_feature_input"]=Counter(k for fmap in t["phoneme_features"] for k in fmap if not k.startswith("place:"))
            activation=defaultdict(float)
            for phone,fmap in zip(phones,t["phoneme_features"]):
                for candidate in self._phoneme_inventory:activation[candidate]+=_cosine(fmap,_phoneme_features(candidate))
            t["trace_phoneme_activation"]=_normalize_map(activation)
            ph_candidates=self._phonological_candidates(phones)
            t["trace_word_activation"]=[{"entry_id":x["entry"].entry_id,"orthography":x["entry"].orthography,
                "activation":x["similarity"]*math.log1p(x["entry"].frequency)} for x in ph_candidates]
            feedback=defaultdict(float,t["trace_phoneme_activation"])
            for x in ph_candidates[:self.config.max_candidates]:
                for phone in x["entry"].phonemes:feedback[phone]+=.05*float(x["similarity"])
            t["trace_feedback"]=_normalize_map(feedback)
            t["trace_competition"]=_normalize_map({k:v*(1-min(.5,sum(feedback.values())-v)) for k,v in feedback.items()})
            cohort=[e for e in self.lexicon if phones and e.phonemes[0]==phones[0]]
            steps=[{"phoneme_index":0,"remaining":[e.entry_id for e in cohort]}]; rp=None
            for i in range(1,len(phones)+1):
                prefix=tuple(phones[:i]); cohort=[e for e in cohort if len(e.phonemes)>=i and e.phonemes[:i]==prefix]
                steps.append({"phoneme_index":i,"remaining":[e.entry_id for e in cohort]})
                if rp is None and len(cohort)==1:rp=i
            t["cohort_trace"]=steps;t["cohort_initial"]=steps[0]["remaining"];t["cohort_recognition_point"]=rp
            t["phonological_candidates"]=ph_candidates
        mark(33,feature_map="aggregated articulatory-feature evidence")
        mark(34,trace_phoneme_nodes=sum(len(t["trace_phoneme_activation"]) for t in work))
        mark(35,trace_word_candidates=sum(len(t["trace_word_activation"]) for t in work))
        mark(36,feedback_gain=.05,feedback_source="active_lexical_candidates")
        mark(37,competition="normalized_lateral_suppression")
        mark(38,cohort_sizes=[len(t["cohort_initial"]) for t in work])
        mark(39,cohort_survivors=[len(t["cohort_trace"][-1]["remaining"]) for t in work])
        mark(40,recognition_points=[t["cohort_recognition_point"] for t in work])

        for t in work:
            shortlist,posterior=self._bayesian_shortlist(t["integrated_pronunciation"],t["phonological_candidates"],context_state)
            t["shortlist"]=shortlist;t["bayesian_posterior"]=posterior
            entropy=-float(np.sum(posterior*np.log(posterior+_EPS))) if posterior.size else 0.0
            t["posterior_entropy"]=entropy/math.log(max(len(posterior),2)) if posterior.size else 1.0
            t["posterior_margin"]=float(posterior[0]-posterior[1]) if posterior.size>1 else float(posterior[0]) if posterior.size else 0.0
            t["lexical_abstain"]=not bool(posterior.size) or float(posterior[0])<self.config.confidence_threshold or float(t["shortlist"][0]["phonological_similarity"])<.5 or t["posterior_margin"]<self.config.ambiguity_margin
        mark(41,shortlist_sizes=[len(t["shortlist"]) for t in work])
        mark(42,noisy_channel="phoneme_edit_distance_with_configured_noise")
        mark(43,prior="smoothed_frequency_exposure")
        mark(44,context_source="caller_context_features")
        mark(45,posterior_equation="P(word|evidence,context) ∝ P(evidence|word)P(word)P(context|word)")
        mark(46,alternatives_preserved=True,noise=self.config.phonological_noise)
        mark(47,temperature=self.config.bayesian_temperature)
        mark(48,confidence_threshold=self.config.confidence_threshold,margin_threshold=self.config.ambiguity_margin)

        for t in work:
            o2s,p2s,triangle,info=self._triangle_for_token(t["graphemes"],t["integrated_pronunciation"])
            t["orthographic_vector"]=_normalize_map(Counter(t["graphemes"]))
            t["phonological_vector"]=_normalize_map(Counter(t["integrated_pronunciation"]))
            t["semantic_feature_space"]=list(self._semantic_inventory);t["triangle_activations"]=info
            t["o_to_s"]=o2s;t["p_to_s"]=p2s;t["triangle_semantic"]=triangle
            best=t["shortlist"][0]["entry"] if t["shortlist"] else None
            direct=dict(best.semantic_features) if best else {}; merged={}
            all_features=set(direct)|set(o2s)|set(p2s)|set(triangle)
            for f in all_features:
                merged[f]=float(np.clip(.35*direct.get(f,0)+.20*o2s.get(f,0)+.20*p2s.get(f,0)+.25*triangle.get(f,0),0,1))
            t["semantic_activation"]=merged
        mark(49,alphabet_dimensions=len(self._alphabet));mark(50,phoneme_dimensions=len(self._phoneme_inventory))
        mark(51,semantic_dimensions=len(self._semantic_inventory));mark(52,association_matrix_shape=list(self._w_op.shape))
        mark(53,association_matrix_shape=list(self._w_os.shape));mark(54,association_matrix_shape=list(self._w_ps.shape))
        mark(55,settling_cycles=5,damping="0.55 previous + 0.45 proposal")
        mark(56,semantic_support=[len(t["semantic_activation"]) for t in work])

        sequence_context=dict(context_state); final_tokens=[]
        for index,t in enumerate(work):
            semantic=t["semantic_activation"]; neighbor=defaultdict(float)
            for ni in (index-1,index+1):
                if 0<=ni<len(work):
                    for f,v in work[ni]["semantic_activation"].items():neighbor[f]+=float(v)*.5
            combined=dict(sequence_context)
            for f,v in neighbor.items():combined[f]=min(1.0,combined.get(f,0)+v)
            t["neighbor_context"]=dict(neighbor); senses=[]
            for item in t["shortlist"]:
                fit=self._context_score(item["entry"],combined)
                senses.append({"entry_id":item["entry"].entry_id,"orthography":item["entry"].orthography,
                    "posterior":float(item["posterior"]*(1+self.config.context_gain*fit)),"context_fit":fit})
            if senses:
                norm=sum(x["posterior"] for x in senses)+_EPS
                for x in senses:x["posterior"]/=norm
                senses.sort(key=lambda x:x["posterior"],reverse=True)
            best=senses[0] if senses else None
            entry=next((e for e in self.lexicon if best and e.entry_id==best["entry_id"]),None)
            t["relation_proposals"]=[]
            if entry and entry.lexical_category in {"noun","verb","adjective","adverb"}:
                t["relation_proposals"].append({"category":entry.lexical_category,"token_index":index,
                    "relation":"ordered_lexical_unit","status":"HEURISTIC_NOT_PARSED_SYNTAX"})
            for f,v in semantic.items():sequence_context[f]=float(np.clip(.8*sequence_context.get(f,0)+.2*v,0,1))
            coherence=_cosine(semantic,combined)
            status="ABSTAIN" if best is None or t["lexical_abstain"] else "RESOLVED"
            t["contextual_senses"]=senses;t["situation_state"]=dict(sequence_context)
            t["semantic_coherence"]=coherence;t["final_candidate"]=best;t["final_status"]=status
            t["explanation"]={"pronunciation_source":t["pronunciation_source"],"route_weights":dict(t["route_weights"]),
                "lexical_candidate_count":len(t["lexical_candidates"]),"bayesian_candidate_count":len(t["shortlist"]),
                "semantic_feature_count":len(semantic),"abstain_reason":"no_supported_lexical_candidate" if best is None else "posterior_below_threshold_or_margin" if t["lexical_abstain"] else None}
            serialized=[]
            for item in t["shortlist"]:
                e=item["entry"];serialized.append({"entry_id":e.entry_id,"orthography":e.orthography,
                    "phonemes":list(e.phonemes),"posterior":item["posterior"],"likelihood":item["likelihood"],
                    "prior":item["prior"],"context_fit":item["context_fit"],"frequency":e.frequency,
                    "verified_by":e.verified_by,"source_reference":e.source_reference})
            final_tokens.append({"index":index,"surface":t["surface"],"prefix_punctuation":t["prefix_punctuation"],
                "suffix_punctuation":t["suffix_punctuation"],"orthography":t["word"],"graphemes":list(t["graphemes"]),
                "orthographic_codes":t["orthographic_codes"],"lexical_candidates":[
                    {"entry_id":x["entry"].entry_id,"orthography":x["entry"].orthography,"similarity":x["similarity"],
                     "exact":x["exact"],"activation":x.get("activation",0)} for x in t["lexical_candidates"]],
                "lexical_pronunciation":list(t["lexical_pronunciation"]),"sublexical_pronunciation":list(t["sublexical_pronunciation"]),
                "g2p_chunks":t["g2p_chunks"],"pronunciation":list(t["integrated_pronunciation"]),
                "pronunciation_source":t["pronunciation_source"],"route_weights":t["route_weights"],
                "pronunciation_confidence":t["pronunciation_confidence"],"syllable_nuclei":t["syllable_nuclei"],
                "stress_pattern":t["stress_pattern"],"phonological_buffer":t["phonological_buffer"],
                "trace_word_activation":t["trace_word_activation"],"cohort_trace":t["cohort_trace"],
                "cohort_recognition_point":t["cohort_recognition_point"],"shortlist":serialized,
                "posterior_entropy":t["posterior_entropy"],"posterior_margin":t["posterior_margin"],
                "lexical_abstain":t["lexical_abstain"],"semantic_activation":t["semantic_activation"],
                "triangle_activations":t["triangle_activations"],"contextual_senses":senses,
                "relation_proposals":t["relation_proposals"],"semantic_coherence":coherence,
                "final_candidate":best,"status":status,"explanation":t["explanation"]})
        mark(57,sequence_token_count=len(final_tokens),lexical_hypotheses_preserved=True)
        mark(58,neighbor_window=1,context_features=len(sequence_context))
        mark(59,context_gain=self.config.context_gain,ambiguous_senses_reweighted=True)
        mark(60,typed_relation_count=sum(len(t["relation_proposals"]) for t in work),syntax_status="heuristic_lexical_categories_only")
        mark(61,situation_state_features=len(sequence_context))
        coherence_values=[t["semantic_coherence"] for t in work]
        mark(62,mean_semantic_coherence=float(np.mean(coherence_values)) if coherence_values else 0)
        mark(63,resolved_count=sum(t["final_status"]=="RESOLVED" for t in work),abstained_count=sum(t["final_status"]=="ABSTAIN" for t in work))
        mark(64,explanations="evidence_ledger_per_token")

        drives={"orthographic":[],"lexical":[],"phonological":[],"auditory_wordform":[],"semantic":[],"control":[]}
        for t in work:
            drives["orthographic"].extend(t["visual_letter_evidence"].values())
            drives["lexical"].extend(t["interactive_activations"].tolist())
            drives["phonological"].extend([1.0]*len(t["integrated_pronunciation"]))
            drives["auditory_wordform"].extend([x["similarity"] for x in t["phonological_candidates"]])
            drives["semantic"].extend(t["semantic_activation"].values())
            drives["control"].extend([t["pronunciation_confidence"],1-t["posterior_entropy"],t["posterior_margin"]])
        mark(65,layer_sizes={k:len(v) for k,v in drives.items()})
        if self.config.simulate_biology:
            lif=LIFConfig(dt_ms=self.config.neural_dt_ms,steps=self.config.neural_steps,tau_m_ms=self.config.membrane_tau_ms,
                refractory_ms=self.config.refractory_ms,input_gain_pa=self.config.input_current_gain_pa,
                recurrent_gain_pa=self.config.recurrent_gain_pa,inhibition_gain_pa=self.config.inhibition_gain_pa)
            biological=BiologicalReadingSimulator(lif).run(drives)
            mark(66,population="orthographic",model="current_based_LIF")
            mark(67,populations=["lexical","phonological"],recurrent_competition=True)
            mark(68,populations=["auditory_wordform","semantic","control"],feedback="bounded_feedforward_proxy")
            layers=biological["layers"]
            mark(69,mean_rates_hz={k:v["mean_firing_rate_hz"] for k,v in layers.items()},
                voltage_unit="mV",current_unit="pA",rate_unit="Hz",time_unit="ms")
            functional=np.asarray([max((t["pronunciation_confidence"] for t in work),default=0),
                max((float(t["final_candidate"]["posterior"]) for t in work if t["final_candidate"] is not None),default=0)])
            neural=np.asarray([float(np.mean(layers["phonological"]["firing_rates_hz"]))/1000,
                float(np.mean(layers["semantic"]["firing_rates_hz"]))/1000])
            residual=float(np.linalg.norm(functional-neural));biological["functional_neural_residual"]=residual
            mark(70,normalized_functional_neural_residual=residual,interpretation="diagnostic_only_not_biological_validation")
        else:
            biological={"status":"DISABLED","scope":"biological simulation disabled by configuration"}
            for number,name in ((66,"orthographic"),(67,"lexical_phonological"),(68,"auditory_semantic"),(69,"measurement"),(70,"residual")):
                spec=OPERATION_REGISTRY[number-1]
                trace.append({"index":number,"operation_id":spec.operation_id,"subsystem":spec.subsystem,
                    "operation":spec.name,"status":"SKIPPED","details":{"reason":"simulate_biology=False","population":name}})
                cursor+=1

        prob_valid=all(not t["bayesian_posterior"].size or np.isclose(float(t["bayesian_posterior"].sum()),1,atol=1e-6) for t in work)
        finite_semantics=all(all(math.isfinite(float(v)) for v in t["semantic_activation"].values()) for t in work)
        trace_prefix=[x["operation_id"] for x in trace]==[x.operation_id for x in OPERATION_REGISTRY[:70]]
        bio_finite=True
        if "layers" in biological:
            for layer in biological["layers"].values():
                bio_finite=bio_finite and math.isfinite(float(layer["mean_firing_rate_hz"])) and math.isfinite(float(layer["mean_membrane_potential_mv"]))
        mark(71,probability_rows_valid=prob_valid,finite_semantics=finite_semantics,biological_finite=bio_finite,
             trace_order_valid=trace_prefix,token_coverage=len(final_tokens)==len(work))
        decoded=" ".join(t["prefix_punctuation"]+(t["final_candidate"]["orthography"] if t["final_candidate"] and t["status"]=="RESOLVED" else t["orthography"])+t["suffix_punctuation"] for t in final_tokens)
        mean_conf=float(np.mean([t["pronunciation_confidence"] for t in work]))
        metadata={"schema":"ore.integrated_reading.v1","registry_id":"ORE-INTEGRATED-READING-72","registry_version":"1.0.0",
            "functional_algorithm":"orthographic_lexicon + interactive_activation + dual_route_G2P + TRACE + Cohort + Shortlist_B + connectionist_triangle",
            "biological_backend":biological.get("model",biological.get("status","unknown")),"lexicon_entry_count":len(self.lexicon),
            "g2p_rule_count":len(self.g2p_rules),"reading_direction":request.reading_direction,"language":request.language,
            "input_source_reference":request.source_reference,"configuration":self.config.to_dict(),
            "elapsed_seconds":float(perf_counter()-started),
            "scientific_scope":"Functional models are algorithmic approximations; LIF is a biologically inspired population model, not a fitted or validated reproduction of human reading.",
            "limitations":["Input is text/tokenized orthography; image OCR and segmentation are upstream.",
                "Default G2P is a small English-oriented demonstration requiring language calibration.",
                "TRACE, Cohort and Shortlist B are approximations, not exact replications.",
                "Semantic features and lexicon pronunciations must be supplied with provenance.",
                "LIF populations are generic, not anatomically mapped or empirically fitted."]}
        mark(72,schema="ore.integrated_reading.v1",decoded_text=decoded,biological_status=biological.get("status","SIMULATED"),trace_length=len(trace)+1)
        complete=len(trace)==72 and [x["operation_id"] for x in trace]==[x.operation_id for x in OPERATION_REGISTRY]
        diagnostics={"valid":bool(prob_valid and finite_semantics and bio_finite and complete),
            "operation_count":len(trace),"operation_trace_complete":complete,"token_count":len(final_tokens),
            "resolved_token_count":sum(t["status"]=="RESOLVED" for t in final_tokens),
            "abstained_token_count":sum(t["status"]=="ABSTAIN" for t in final_tokens),
            "mean_pronunciation_confidence":mean_conf,
            "mean_semantic_coherence":float(np.mean(coherence_values)) if coherence_values else 0.0,
            "probability_rows_valid":prob_valid,"finite_semantics":finite_semantics,"biological_finite":bio_finite,
            "warnings":warnings}
        return ReadingResult(request.sequence_id,original,normalized,final_tokens,decoded,dict(sequence_context),
            biological,trace,diagnostics,metadata)
