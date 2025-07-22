# Neural Mechanisms of Perceptual Confidence

This project investigates how confidence emerges from the dynamics of neural populations during perceptual decision-making, using biologically plausible spiking neural networks and established psychophysical paradigms.

Based on established neuroscience literature (Shadlen & Newsome, 2001; Kiani & Shadlen, 2009; Pouget et al., 2016), this project tests the hypothesis that:

1. **Confidence emerges from decision dynamics**: Not just final output probabilities, but from the temporal evolution of evidence accumulation
2. **Neural variability drives uncertainty**: Trial-to-trial variability in neural responses correlates with subjective confidence
3. **Competing attractors generate choice**: Decision-making emerges from competition between neural populations
4. **Time-to-decision predicts confidence**: Faster decisions (shorter RTs) correlate with higher confidence

## Experimental Design

### Task: Motion Coherence Decision
- **Stimulus**: Random dot kinematogram with varying motion coherence (0-51.2%)
- **Choice**: Left vs Right motion direction
- **Confidence**: Binary confidence judgment (high/low confidence)
- **Context**: Rule switches between motion direction and motion speed judgments

### Model Architecture
- **Spiking Neural Network**: Leaky integrate-and-fire neurons
- **Populations**: Sensory (MT/MST), Decision (LIP), Confidence (area X)
- **Connectivity**: Biologically constrained (Dale's principle, realistic delays)
- **Time constants**: Realistic membrane and synaptic dynamics

### Analysis Methods
1. **Decision dynamics**: Evidence accumulation trajectories
2. **Confidence mechanisms**: Population variance, time-to-threshold, choice predictive activity
3. **Neural-behavioral correlations**: Spike count correlations with choice and confidence
4. **Dynamical systems**: Attractor landscapes, flow fields, stability analysis

## Scientific Predictions

1. Higher coherence → faster decisions → higher confidence
2. Neural variance anti-correlates with confidence
3. Confidence correlates with decision boundary distance at choice time
4. Context switches slow decisions and reduce confidence

## Validation
- Compare to human psychophysical data (Kiani & Shadlen, 2009)
- Benchmark against drift-diffusion models
- Test specific neural predictions from monkey electrophysiology

---
