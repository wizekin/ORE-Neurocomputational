"""Current-based leaky integrate-and-fire reference populations for lexical-context processing.

This is a generic rate/neuron research model, not an anatomically fitted or validated brain model.
"""
from __future__ import annotations
from typing import Mapping, Sequence
import math
import numpy as np
from .contracts import LexicalContextConfig

POPULATIONS = ("orthographic","lexical","morphological","phonological","semantic","context_predictive")

class LexicalContextLIFSimulator:
    _CONNECTIONS = {
        "orthographic":{"lexical":0.18,"morphological":0.04},
        "lexical":{"orthographic":0.08,"phonological":0.14,"semantic":0.12},
        "morphological":{"lexical":0.12,"semantic":0.16},
        "phonological":{"lexical":0.12,"semantic":0.06},
        "semantic":{"lexical":0.10,"context_predictive":0.18},
        "context_predictive":{"lexical":0.14,"semantic":0.12},
    }
    def __init__(self, config: LexicalContextConfig):
        self.config=config; n=config.population_size
        self.voltage_mv={p:np.full(n,config.rest_mv,dtype=float) for p in POPULATIONS}
        self.refractory_ms={p:np.zeros(n,dtype=float) for p in POPULATIONS}
        self.adaptation={p:np.zeros(n,dtype=float) for p in POPULATIONS}
        self.spike_counts={p:np.zeros(n,dtype=np.int64) for p in POPULATIONS}
        self.activity={p:0.0 for p in POPULATIONS}; self.last_current_pa={p:np.zeros(n) for p in POPULATIONS}
        self.time_ms=0.0
    def _advance(self, drives: Mapping[str,float]) -> None:
        c=self.config; global_activity=float(np.mean(list(self.activity.values())))
        new_v={}; new_ref={}; new_adapt={}; new_activity={}
        for name in POPULATIONS:
            drive=float(np.clip(drives.get(name,0.0),0.0,1.0))
            afferent=sum(weight*self.activity[source] for source,weight in self._CONNECTIONS[name].items())
            current_scalar=(c.current_gain_pa*drive+c.recurrent_gain_pa*(afferent+self.activity[name])
                            -c.inhibition_gain_pa*global_activity)
            current=np.full(c.population_size,current_scalar)-c.adaptation_current_gain_pa*self.adaptation[name]
            self.last_current_pa[name]=current.copy()
            voltage=self.voltage_mv[name].copy(); refractory=self.refractory_ms[name].copy()
            adaptation=self.adaptation[name].copy()
            active=refractory<=0.0
            voltage[~active]=c.reset_mv; refractory[~active]=np.maximum(0.0,refractory[~active]-c.dt_ms)
            dv=(c.dt_ms/c.membrane_tau_ms)*(c.rest_mv-voltage+c.membrane_resistance_mohm*current*0.001)
            voltage[active]+=dv[active]
            spiked=active & (voltage>=c.threshold_mv)
            self.spike_counts[name][spiked]+=1; voltage[spiked]=c.reset_mv; refractory[spiked]=c.refractory_ms
            decay=math.exp(-c.dt_ms/max(c.membrane_tau_ms*4,c.dt_ms))
            adaptation*=decay; adaptation[spiked]+=1.0
            fraction=float(np.count_nonzero(spiked))/c.population_size
            new_activity[name]=math.exp(-c.dt_ms/max(c.membrane_tau_ms,c.dt_ms))*self.activity[name]+(1-math.exp(-c.dt_ms/max(c.membrane_tau_ms,c.dt_ms)))*fraction
            new_v[name]=voltage; new_ref[name]=refractory; new_adapt[name]=adaptation
        self.voltage_mv.update(new_v); self.refractory_ms.update(new_ref); self.adaptation.update(new_adapt)
        self.activity.update(new_activity); self.time_ms+=c.dt_ms
    def step(self, operation_id: str, token_drives: Sequence[Mapping[str,float]]) -> dict[str,object]:
        if not token_drives: token_drives=({},)
        for drives in token_drives:
            unknown=set(drives)-set(POPULATIONS)
            if unknown: raise KeyError(f"unknown populations: {sorted(unknown)}")
            if any(not math.isfinite(float(v)) for v in drives.values()): raise ValueError("drives must be finite")
            for _ in range(self.config.steps_per_operation): self._advance(drives)
        snapshot=self.snapshot()
        return {"operation_id":operation_id,"status":"simulated","simulated_time_ms":snapshot["simulated_time_ms"],
                "population_activity":dict(self.activity),
                "mean_firing_rates_hz":{k:v["mean_firing_rate_hz"] for k,v in snapshot["layers"].items()}}
    def snapshot(self) -> dict[str,object]:
        elapsed=max(self.time_ms/1000.0,1e-9); layers={}
        for p in POPULATIONS:
            layers[p]={"mean_voltage_mv":float(np.mean(self.voltage_mv[p])),
                "final_voltage_mv":self.voltage_mv[p].tolist(),
                "mean_current_pa":float(np.mean(self.last_current_pa[p])),
                "spike_count":int(np.sum(self.spike_counts[p])),
                "mean_firing_rate_hz":float(np.sum(self.spike_counts[p])/(self.config.population_size*elapsed)),
                "activity_fraction":float(self.activity[p]),"mean_adaptation":float(np.mean(self.adaptation[p]))}
        finite=all(np.isfinite(a).all() for d in (self.voltage_mv,self.last_current_pa,self.adaptation) for a in d.values())
        return {"enabled":True,"model":"generic_recurrent_current_based_LIF_reference","layers":layers,
            "layer_order":list(POPULATIONS),"simulated_time_ms":float(self.time_ms),"numerically_valid":bool(finite),
            "units":{"membrane_voltage":"mV","synaptic_current":"pA","firing_rate":"Hz","simulated_time":"ms","spike_count":"count"},
            "scope":"Generic LIF populations with recurrent coupling, inhibition and adaptation; not a fitted, anatomically mapped or empirically validated model of human language processing."}
