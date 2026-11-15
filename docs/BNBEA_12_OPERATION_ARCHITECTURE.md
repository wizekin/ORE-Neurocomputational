# BNBEA — Brain Networks and Biological Evidence Algorithm

**Python package:** ore_brain_networks  
**Registry:** BNBEA-001 through BNBEA-012

BNBEA combines visual and auditory feature processing, ordered grapheme/orthographic representations, recurrent language-network integration, explicit neural population dynamics, and experimental measurement/validation models.

The architecture deliberately separates three kinds of results:

1. **Functional representation:** image features, grapheme candidates, orthographic sequences, lexical alternatives, acoustic features and language-network states.
2. **Biological reference:** simulated current-based leaky integrate-and-fire (LIF) populations, membrane voltages, synaptic currents, spikes, firing rates and recurrent activity.
3. **Experimental prediction/validation:** EEG/MEG forward and inverse estimates, BOLD proxies, virtual lesion/TMS effects and comparisons with independently supplied human observations.

These mathematical operations are computational hypotheses. The code does not claim that the brain literally computes Gabor convolutions, softmax distributions, graph matrix products or the exact numerical routines implemented here.

## Operation registry

| ID | Operation | Functional and mathematical implementation | Biological reference or evidence role |
|---|---|---|---|
| BNBEA-001 | Gabor-inspired early visual processing | Grayscale and contrast normalization; oriented Gabor convolution; quadrature energy; rectification; divisive normalization | Retinal/LGN contrast pathways and V1 orientation/spatial-frequency selectivity |
| BNBEA-002 | Neural population-style grapheme prototypes | Candidate-region segmentation; feature vectors; Mahalanobis-like prototype distance; radial-basis tuning; softmax competition; ordered positions; EMA learning | Distributed feature-selective population coding; prototype bank is an engineering abstraction |
| BNBEA-003 | Explicit simulated neural dynamics | Synaptic-current summation; membrane differential equation; Euler integration; threshold/reset/refractory; recurrent population rates | Current-based LIF reference model with explicit voltage, current, time and spike state |
| BNBEA-004 | Biological hierarchical visual populations | Retina/LGN, V1, V2, V4, ventral temporal and dorsal spatial maps; attention; feedback | Hierarchical receptive-field complexity and recurrent visual processing; not a complete anatomical wiring diagram |
| BNBEA-005 | Biological VWFA hierarchy | Ordered posterior graphemes, overlapping n-grams, anterior lexical candidates, edit similarity and lexical competition | Posterior-to-anterior ventral occipitotemporal/VWFA hierarchy; detailed subregional functions remain hypotheses |
| BNBEA-006 | Anatomical IFG/MTG/ATL network | Recurrent graph propagation through VWFA, MTG, IFG, ATL, STG and AG; semantic/context vector integration | Distributed language network; no one-region/one-function assumption |
| BNBEA-007 | Auditory dorsal and ventral streams | STFT, power spectrum, ERB-like cochlear filterbank, temporal envelope and dorsal/ventral feature projections | Auditory comprehension and auditory-motor pathways; no transcript is fabricated |
| BNBEA-008 | EEG/MEG signal modeling | Forward projection y=L s+noise; regularized inverse estimate; ERP averaging; time-frequency power; waveform error | Sensor measurements are geometry-dependent sums of neural currents; generic lead fields are explicitly toy models |
| BNBEA-009 | fMRI comparison | Regional activity; double-gamma HRF convolution BOLD=h*n; least-squares GLM; RDM and correlation | BOLD is an indirect delayed hemodynamic response, not a direct spike or voltage measurement |
| BNBEA-010 | Lesion-based validation | Fractional region/edge attenuation; activity changes; lesion-symptom OLS; weighted graph global efficiency | Virtual lesion counterfactuals; necessity claims require independent patient data |
| BNBEA-011 | TMS-based validation | Unitless time-dependent perturbation; active/sham/control contrasts; observed-effect discrepancy | Software-only intervention simulation; no physical or clinical stimulation settings are prescribed |
| BNBEA-012 | Independent human-reading comparison | Accuracy, MAE/RMSE, waveform MSE, RDM correlation, subject-disjoint partitions, bootstrap intervals and ablation deltas | Independent human measurements are required; the software demonstration does not establish biological validity |

## Mathematical reference equations

The implementation includes the following mathematical families:

- **Visual filtering:** R = I * g, with quadrature energy E = sqrt(Re² + Ro²).
- **Population tuning:** a_i = exp(-d_i² / (2 sigma_i²)), followed by normalized competition.
- **Membrane dynamics:** C_m dV/dt = -g_L(V-E_L) + I(t); a forward-Euler approximation updates voltage at each time step.
- **Recurrent population dynamics:** tau dx/dt = -x + phi(Wx + I), represented as coupled, damped population states.
- **Network propagation:** x[t+1] = phi(Ax[t] + Bu[t]).
- **EEG/MEG forward model:** y(t) = L s(t) + epsilon; the inverse solution uses Tikhonov/minimum-norm regularization.
- **Hemodynamic transformation:** BOLD(t) = h(t) * n(t).
- **Lesion effects:** x'_v = (1-rho_v)x_v and W'ij = (1-rho_ij)Wij.
- **TMS virtual perturbation:** x_TMS(t) = x(t) + P(t), or x_TMS(t) = (1-kappa P(t))x(t).
- **Validation:** accuracy, MAE, RMSE, waveform MSE, RDM correlation and ablation delta loss.

Equations are model choices and not claims of uniquely identified neural calculations. Parameters are configurable engineering values, not universal physiological constants.

## Run the package

Create a request from an image, optional upstream orthographic sequence and optional audio waveform:

    import numpy as np
    from ore_brain_networks import BNBEAProcessor, BNBEARequest, LexicalEntry

    request = BNBEARequest(
        sample_id="reading-001",
        visual_image=np.zeros((64, 96)),
        orthographic_sequence=("c", "a", "t"),
        lexicon=(LexicalEntry("cat", frequency=30.0),),
        semantic_features={"ANIMAL": 1.0},
        context_features={"PET": 0.7},
    )
    result = BNBEAProcessor().process(request)
    assert len(result.operation_trace) == 12

The command-line smoke run is:

    python -m ore_brain_networks

To run the focused tests:

    python -m pytest tests/test_brain_networks.py -q

## Input and output boundaries

- A grayscale or RGB image can enter Operation 1. Operation 2 produces candidate stroke regions and can classify them only when a compatible labelled GraphemePrototype inventory is supplied.
- An upstream orthographic sequence can be passed via orthographic_sequence. This package does not claim to replace a trained OCR system.
- Lexical candidate selection uses only the caller-supplied lexicon. It does not invent dictionary entries.
- An audio waveform produces frequency/time and phonological feature vectors. It does not generate a transcript or speech labels.
- EEG/MEG experiments may provide lead fields and aligned recordings. Otherwise the package generates a synthetic toy projection marked as unsuitable for source localization.
- fMRI comparisons require compatible timing, design matrices and recorded patterns. Generated BOLD is only an HRF-convolved activity proxy.
- Lesion and TMS modules operate on model state only. They do not act on people or recommend clinical or research stimulation protocols.
- Independent human validation waits for actual measurements and IDs; it does not fabricate participant data or biological agreement.

## SQLite annotations

Every package module, class and callable has an explanatory Python docstring. The repository's existing AST catalogue indexes these descriptions in the SQLite annotations table. BNBEAProcessor synchronizes the repository catalogue by default.

    from ore_brain_networks import register_bnbea_annotations
    report = register_bnbea_annotations()
    print(report["database_path"])

The default database is ore_logs/ore_logger.sqlite3. Set ORE_LOGGER_DB or pass database_path to choose another file. The SQLite runtime database is generated locally and is not committed to Git; source docstrings remain version-controlled and the logger can refresh the index.

## Scientific limitations

- Gabor filters, feature hashing, prototype distances, graph weights and LIF states are transparent computational analogies, not direct recordings of brain activity.
- The LIF network is a compact population reference, not an anatomically fitted cellular model or complete connectome.
- Synthetic EEG/MEG lead fields are not participant head models. Real source modeling requires valid sensor geometry, source locations, conductivity/head models and preprocessing.
- BOLD is a hemodynamic proxy. Meaningful scanner comparison requires acquisition timing, preprocessing, design matrices and vascular assumptions.
- Lesion and TMS computations are virtual interventions; they do not prove causal necessity or reproduce every effect of human brain injury or stimulation.
- Biological validity requires independent behavioral and neural measurements, suitable baselines, subject/stimulus-disjoint evaluation, uncertainty reporting and comparison to simpler alternatives.
- Passing software tests demonstrates implementation behavior, not biological accuracy.
