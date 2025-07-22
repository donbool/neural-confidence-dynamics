# dynamics.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model import ContextRNN
from generate_data import generate_trials
from train import prepare_dataloader

def extract_hidden_trajectories(model, dataloader):
    """
    Extract hidden state vectors from each input sample.
    """
    model.eval()
    all_hidden = []

    with torch.no_grad():
        for X_batch, _ in dataloader:
            _, h = model(X_batch)
            all_hidden.append(h.squeeze(0).numpy())

    return np.vstack(all_hidden)

def plot_2D_dynamics(hidden, title="RNN Hidden Dynamics (PCA)"):
    """
    Visualize hidden states projected into 2D space.
    """
    pca = PCA(n_components=2)
    projected = pca.fit_transform(hidden)

    plt.figure(figsize=(8, 6))
    plt.scatter(projected[:, 0], projected[:, 1], alpha=0.6, s=15)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load data and model
    trials = generate_trials(n_trials=500, noise_std=0.25, structured_noise=True)
    dataloader = prepare_dataloader(trials)

    model = ContextRNN(input_size=3, hidden_size=64, output_size=2)
    model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))

    # Extract and plot
    hidden = extract_hidden_trajectories(model, dataloader)
    plot_2D_dynamics(hidden)
