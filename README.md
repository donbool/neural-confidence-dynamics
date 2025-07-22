# emergent confidence and context-dependent dynamics in noisy recurrent neural systems


this project simulates a context-dependent decision-making task using a noisy recurrent neural network (RNN), designed to explore how confidence-like signals and dynamic representations emerge from internal neural activity under structured noise. the model mimicks aspects of biological decision-making, such as: context switches, stimulus ambiguity, and trial-to-trial variability. 


this is just a computational experiment aligned with principles from ML and cognitive neuroscience.

---

## goals

- simulate a flexible decision-making task with noisy inputs and shifting context rules  
- train an RNN to solve the task and generalize across contexts  
- analyze internal dynamics to identify latent correlations of confidence and belief  
- study how context remapping and structured noise affect neural representations

---

## looking into

- noisy stimulus encoding and rule-based context switching  
- confidence as an emergent property from internal dynamics (e.g., entropy, distance to boundary) rather than being the output  
- latent space visualization using PCA, t-SNE, or UMAP idk yet
- possibly going to do dynamical systems analysis via fixed points and stability

---

## what u shall see

- accuracy and confidence metrics across trial conditions  
- visualization of RNN state trajectories colored by stimulus, context, and trial difficulty  
- confidence-accuracy correlation plots  
- attractor analysis and flow field visualizations of RNN dynamics
