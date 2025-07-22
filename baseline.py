# baseline.py

import torch
import torch.nn as nn
import numpy as np
from model import ContextRNN
from generate_data import generate_trials
from train import prepare_dataloader, train_model, save_model
import matplotlib.pyplot as plt


class FeedforwardBaseline(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=2):
        """
        Feedforward baseline model for comparison with RNN.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        # Take only the last timestep for feedforward
        if len(x.shape) == 3:
            x = x[:, -1, :]  # (batch, seq_len, features) -> (batch, features)
        
        logits = self.layers(x)
        # Return dummy hidden state for compatibility
        hidden = torch.zeros(1, x.size(0), 64)
        return logits, hidden


def compare_models():
    """
    Compare RNN vs Feedforward baseline across different noise levels.
    """
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    rnn_accuracies = []
    ff_accuracies = []
    
    for noise in noise_levels:
        print(f"\nTesting noise level: {noise}")
        
        # Generate test data
        test_trials = generate_trials(n_trials=500, noise_std=noise, seq_len=5, seed=42)
        test_loader = prepare_dataloader(test_trials, use_sequences=True, batch_size=32)
        
        # Test RNN
        try:
            rnn_model = ContextRNN(input_size=3, hidden_size=64, output_size=2, rnn_type="gru")
            rnn_model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))
            rnn_acc = evaluate_model(rnn_model, test_loader)
            rnn_accuracies.append(rnn_acc)
        except FileNotFoundError:
            print("No trained RNN model found. Training baseline only.")
            rnn_accuracies.append(0)
        
        # Train and test Feedforward baseline
        train_trials = generate_trials(n_trials=1000, noise_std=noise, seq_len=5, seed=123)
        train_loader = prepare_dataloader(train_trials, use_sequences=True, batch_size=32)
        
        ff_model = FeedforwardBaseline(input_size=3, hidden_size=64, output_size=2)
        ff_model = train_model(ff_model, train_loader, num_epochs=20, lr=1e-3)
        ff_acc = evaluate_model(ff_model, test_loader)
        ff_accuracies.append(ff_acc)
        
        print(f"  RNN Accuracy: {rnn_acc:.3f}")
        print(f"  Feedforward Accuracy: {ff_acc:.3f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, rnn_accuracies, 'o-', label='RNN', linewidth=2, markersize=8)
    plt.plot(noise_levels, ff_accuracies, 's-', label='Feedforward', linewidth=2, markersize=8)
    plt.xlabel('Noise Level (σ)')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison: RNN vs Feedforward Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.4, 1.0)
    plt.tight_layout()
    plt.show()
    
    return {
        'noise_levels': noise_levels,
        'rnn_accuracies': rnn_accuracies,
        'ff_accuracies': ff_accuracies
    }


def evaluate_model(model, dataloader):
    """
    Evaluate model accuracy on test data.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            logits, _ = model(X_batch)
            predicted = logits.argmax(dim=1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    
    return correct / total


def analyze_context_sensitivity():
    """
    Test how well models handle context switches vs single contexts.
    """
    print("\nAnalyzing context sensitivity...")
    
    # Single context trials
    single_trials = generate_trials(n_trials=500, context_switch=False, seq_len=5, seed=42)
    single_loader = prepare_dataloader(single_trials, use_sequences=True)
    
    # Context switch trials  
    switch_trials = generate_trials(n_trials=500, context_switch=True, seq_len=5, seed=42)
    switch_loader = prepare_dataloader(switch_trials, use_sequences=True)
    
    try:
        # Test RNN
        rnn_model = ContextRNN(input_size=3, hidden_size=64, output_size=2, rnn_type="gru")
        rnn_model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))
        
        rnn_single_acc = evaluate_model(rnn_model, single_loader)
        rnn_switch_acc = evaluate_model(rnn_model, switch_loader)
        
        print(f"RNN - Single Context: {rnn_single_acc:.3f}")
        print(f"RNN - Context Switch: {rnn_switch_acc:.3f}")
        print(f"RNN - Switch Performance Ratio: {rnn_switch_acc/rnn_single_acc:.3f}")
        
    except FileNotFoundError:
        print("No trained RNN model found for context analysis.")
    
    # Train feedforward on mixed data
    mixed_trials = generate_trials(n_trials=1000, context_switch=True, seq_len=5, seed=123)
    train_loader = prepare_dataloader(mixed_trials, use_sequences=True)
    
    ff_model = FeedforwardBaseline()
    ff_model = train_model(ff_model, train_loader, num_epochs=20, lr=1e-3)
    
    ff_single_acc = evaluate_model(ff_model, single_loader)
    ff_switch_acc = evaluate_model(ff_model, switch_loader)
    
    print(f"Feedforward - Single Context: {ff_single_acc:.3f}")
    print(f"Feedforward - Context Switch: {ff_switch_acc:.3f}")
    print(f"Feedforward - Switch Performance Ratio: {ff_switch_acc/ff_single_acc:.3f}")


if __name__ == "__main__":
    # Run baseline comparison
    results = compare_models()
    
    # Analyze context switching
    analyze_context_sensitivity()