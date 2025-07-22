# neural mechs of perceptual confidence

failed, useless lol


investigating how confidence emerges from the dynamics of neural populations during perceptual decision-making, using biologically plausible spiking neural networks and established psychophysical paradigms.

based on (Shadlen & Newsome, 2001; Kiani & Shadlen, 2009; Pouget et al., 2016), testing:

1. confidence emerges from decision dynamics: Not just final output probabilities, but from the temporal evolution of evidence accumulation
2. neural variability drives uncertainty: trial-to-trial variability in neural responses correlates with subjective confidence
3. competing attractors generate choice: decision-making emerges from competition between neural populations
4. time-to-decision predicts confidence: faster decisions (shorter RTs) correlate with higher confidence

## design

### motion coherence detection
- random dot kinematogram with varying motion coherence (0-51.2%)
- left v right motion direction
- binary confidence judgment (high/low confidence)
- rule switches between motion direction and motion speed judgments

### model arch
- spiking neural nets
- populations: sensory (MT/MST), decision (LIP), confidence (area X)
- biologically constrained (dale's principle, realistic delays)
- time constraints via realistic membrane and synaptic dynamics

### analysis
- evidence accumulation trajectories
- population variance, time-to-threshold, choice predictive activity
- spike count correlations with choice and confidence
- attractor landscapes, flow fields, stability analysis

## predictions

- higher coherence → faster decisions → higher confidence
- neural variance anti-correlates with confidence
- confidence correlates with decision boundary distance at choice time
- context switches slow decisions and reduce confidence

## val
- compare to human psychophysical data (Kiani & Shadlen, 2009)
- benchmark against drift-diffusion models
