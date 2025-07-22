# generate_data.py

import numpy as np
import random


def generate_trials(
    n_trials=1000,
    noise_std=0.2,
    structured_noise=False,
    contexts=("A", "B"),
    seed=None
):
    """
    Generate simulated decision-making trials with noisy stimuli and binary context.

    Args:
        n_trials (int): Number of total trials to generate.
        noise_std (float): Standard deviation of the Gaussian noise.
        structured_noise (bool): If True, add shared noise across trials in a batch.
        contexts (tuple): Context labels ("A" or "B").
        seed (int): Optional random seed for reproducibility.

    Returns:
        dict: Trial dictionary with keys:
              'stimulus', 'context', 'input', 'label'
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    trials = {
        "stimulus": [],
        "context": [],
        "input": [],
        "label": []
    }

    for _ in range(n_trials):
        # Generate stimulus in range [-1, 1]
        stim = np.random.uniform(-1, 1)

        # Select a context randomly
        ctx = random.choice(contexts)

        # Apply noise
        noise = np.random.normal(0, noise_std)
        shared_noise = np.random.normal(0, noise_std) if structured_noise else 0
        observed = stim + noise + shared_noise

        # Context cue as one-hot
        context_vector = [1, 0] if ctx == "A" else [0, 1]

        # Determine label based on context rule
        if ctx == "A":
            label = int(stim > 0)
        else:
            label = int(stim < 0)

        trials["stimulus"].append(stim)
        trials["context"].append(context_vector)
        trials["input"].append([observed] + context_vector)
        trials["label"].append(label)

    return trials


# sample usage
if __name__ == "__main__":
    data = generate_trials(n_trials=5, seed=42)
    for i in range(5):
        print(f"Trial {i+1}: input={data['input'][i]}, label={data['label'][i]}")
