# analyze.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ContextRNN
from generate_data import generate_trials
from train import prepare_dataloader
from sklearn.decomposition import PCA

def get_hidden_activations(model, dataloader):
    """
    Collect hidden states and confidence values from the model.
    """
    model.eval()
    all_hidden = []
    all_labels = []
    all_confidence = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            logits, h = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            confidence = torch.abs(probs[:, 0] - probs[:, 1])  # distance from decision boundary

            all_hidden.append(h.squeeze(0).numpy())
            all_labels.append(y_batch.numpy())
            all_confidence.append(confidence.numpy())

    return (
        np.vstack(all_hidden),
        np.concatenate(all_labels),
        np.concatenate(all_confidence)
    )

def plot_pca(hidden, confidence):
    """
    Plot 2D PCA of hidden states colored by confidence.
    """
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(hidden)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced[:, 0], reduced[:, 1],
        c=confidence, cmap='coolwarm', alpha=0.7
    )
    plt.colorbar(scatter, label='Confidence (|p0 - p1|)')
    plt.title("Hidden State Trajectories (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    trials = generate_trials(n_trials=500, noise_std=0.25, structured_noise=True, seed=42)
    dataloader = prepare_dataloader(trials)

    model = ContextRNN(input_size=3, hidden_size=64, output_size=2, rnn_type="gru")
    model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))

    hidden, labels, confidence = get_hidden_activations(model, dataloader)
    plot_pca(hidden, confidence)
