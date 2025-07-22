# generate_data.py

import numpy as np
import random


def generate_trials(
    n_trials=1000,
    noise_std=0.2,
    structured_noise=False,
    contexts=("A", "B"),
    seq_len=5,
    context_switch=False,
    seed=None
):
    """
    Generate simulated decision-making trials with noisy stimuli and binary context.

    Args:
        n_trials (int): Number of total trials to generate.
        noise_std (float): Standard deviation of the Gaussian noise.
        structured_noise (bool): If True, add shared noise across trials in a batch.
        contexts (tuple): Context labels ("A" or "B").
        seq_len (int): Number of timesteps per trial.
        context_switch (bool): If True, allow context to switch mid-trial.
        seed (int): Optional random seed for reproducibility.

    Returns:
        dict: Trial dictionary with keys:
              'stimulus', 'context', 'input', 'label', 'sequence'
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    trials = {
        "stimulus": [],
        "context": [],
        "input": [],
        "label": [],
        "sequence": []
    }

    for _ in range(n_trials):
        # Generate base stimulus
        stim = np.random.uniform(-1, 1)
        
        # Initialize sequence data
        sequence_inputs = []
        sequence_contexts = []
        ctx = random.choice(contexts)
        
        for t in range(seq_len):
            # Context switch mid-trial if enabled
            if context_switch and t == seq_len // 2:
                ctx = "B" if ctx == "A" else "A"
            
            # Apply noise at each timestep
            noise = np.random.normal(0, noise_std)
            shared_noise = np.random.normal(0, noise_std) if structured_noise else 0
            observed = stim + noise + shared_noise
            
            # Context cue as one-hot
            context_vector = [1, 0] if ctx == "A" else [0, 1]
            
            sequence_inputs.append([observed] + context_vector)
            sequence_contexts.append(context_vector)
        
        # Final label based on last context
        if ctx == "A":
            label = int(stim > 0)
        else:
            label = int(stim < 0)
        
        trials["stimulus"].append(stim)
        trials["context"].append(sequence_contexts[-1])  # Final context
        trials["input"].append([sequence_inputs[-1][0]] + sequence_contexts[-1])  # Final input for backward compatibility
        trials["label"].append(label)
        trials["sequence"].append(sequence_inputs)  # Full sequence

    return trials


# sample usage
if __name__ == "__main__":
    data = generate_trials(n_trials=5, seq_len=3, context_switch=True, seed=42)
    for i in range(5):
        print(f"Trial {i+1}: sequence={data['sequence'][i]}, label={data['label'][i]}")
