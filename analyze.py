# analyze.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ContextRNN
from generate_data import generate_trials
from train import prepare_dataloader
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

def get_hidden_activations(model, dataloader):
    """
    Collect hidden states, predictions, and confidence values from the model.
    """
    model.eval()
    all_hidden = []
    all_labels = []
    all_predictions = []
    all_confidence_margin = []
    all_confidence_entropy = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            logits, h = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            # Confidence as margin (distance from decision boundary)
            confidence_margin = torch.abs(probs[:, 0] - probs[:, 1])
            
            # Confidence as negative entropy (higher = more confident)
            confidence_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

            all_hidden.append(h.squeeze(0).numpy())
            all_labels.append(y_batch.numpy())
            all_predictions.append(preds.numpy())
            all_confidence_margin.append(confidence_margin.numpy())
            all_confidence_entropy.append(confidence_entropy.numpy())

    return {
        'hidden': np.vstack(all_hidden),
        'labels': np.concatenate(all_labels),
        'predictions': np.concatenate(all_predictions),
        'confidence_margin': np.concatenate(all_confidence_margin),
        'confidence_entropy': np.concatenate(all_confidence_entropy)
    }

def plot_pca(hidden, confidence, confidence_type='margin'):
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
    label = f'Confidence ({confidence_type})'
    plt.colorbar(scatter, label=label)
    plt.title(f"Hidden State Trajectories (PCA) - {confidence_type.title()}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_confidence_accuracy_correlation(results):
    """
    Analyze correlation between confidence and accuracy.
    """
    accuracy = (results['predictions'] == results['labels']).astype(float)
    
    # Correlations
    r_margin, p_margin = pearsonr(results['confidence_margin'], accuracy)
    r_entropy, p_entropy = pearsonr(-results['confidence_entropy'], accuracy)  # negative entropy for intuitive direction
    
    print(f"Confidence-Accuracy Correlations:")
    print(f"  Margin:  r={r_margin:.3f}, p={p_margin:.3f}")
    print(f"  Entropy: r={r_entropy:.3f}, p={p_entropy:.3f}")
    
    # Plot confidence vs accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Binned accuracy by confidence
    def plot_binned_accuracy(ax, confidence, title):
        bins = np.linspace(confidence.min(), confidence.max(), 10)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accs = []
        
        for i in range(len(bins)-1):
            mask = (confidence >= bins[i]) & (confidence < bins[i+1])
            if mask.sum() > 0:
                bin_accs.append(accuracy[mask].mean())
            else:
                bin_accs.append(0)
        
        ax.plot(bin_centers, bin_accs, 'o-')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.grid(True)
    
    plot_binned_accuracy(ax1, results['confidence_margin'], 'Margin Confidence')
    plot_binned_accuracy(ax2, -results['confidence_entropy'], 'Entropy Confidence')
    
    plt.tight_layout()
    plt.show()
    
    return {'margin_corr': r_margin, 'entropy_corr': r_entropy}

if __name__ == "__main__":
    trials = generate_trials(n_trials=500, noise_std=0.25, structured_noise=True, seq_len=5, seed=42)
    dataloader = prepare_dataloader(trials, use_sequences=True)

    model = ContextRNN(input_size=3, hidden_size=64, output_size=2, rnn_type="gru")
    model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))

    results = get_hidden_activations(model, dataloader)
    
    # Plot PCA with both confidence measures
    plot_pca(results['hidden'], results['confidence_margin'], 'margin')
    plot_pca(results['hidden'], results['confidence_entropy'], 'entropy')
    
    # Analyze confidence-accuracy correlation
    correlations = analyze_confidence_accuracy_correlation(results)
