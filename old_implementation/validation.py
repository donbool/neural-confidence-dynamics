# validation.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ContextRNN
from baseline import FeedforwardBaseline
from generate_data import generate_trials
from train import prepare_dataloader, train_model
from analyze import get_hidden_activations, analyze_confidence_accuracy_correlation
import pandas as pd
from scipy import stats
import seaborn as sns


def run_multi_seed_validation(n_seeds=5, noise_levels=[0.1, 0.3, 0.5]):
    """
    Run experiments across multiple random seeds for statistical validation.
    """
    print(f"Running validation across {n_seeds} seeds and {len(noise_levels)} noise levels...")
    
    results = {
        'seed': [],
        'noise_level': [],
        'model_type': [],
        'accuracy': [],
        'confidence_margin_corr': [],
        'confidence_entropy_corr': []
    }
    
    for seed in range(n_seeds):
        print(f"\nSeed {seed + 1}/{n_seeds}")
        
        for noise in noise_levels:
            print(f"  Noise level: {noise}")
            
            # Generate data
            train_trials = generate_trials(
                n_trials=1000, noise_std=noise, seq_len=5, 
                context_switch=True, seed=seed
            )
            test_trials = generate_trials(
                n_trials=300, noise_std=noise, seq_len=5,
                context_switch=True, seed=seed + 1000
            )
            
            train_loader = prepare_dataloader(train_trials, use_sequences=True, batch_size=32)
            test_loader = prepare_dataloader(test_trials, use_sequences=True, batch_size=32)
            
            # Train and test RNN
            rnn_model = ContextRNN(input_size=3, hidden_size=64, output_size=2, rnn_type="gru")
            rnn_model = train_model(rnn_model, train_loader, num_epochs=15, lr=1e-3)
            
            rnn_results = evaluate_model_comprehensive(rnn_model, test_loader)
            
            results['seed'].append(seed)
            results['noise_level'].append(noise)
            results['model_type'].append('RNN')
            results['accuracy'].append(rnn_results['accuracy'])
            results['confidence_margin_corr'].append(rnn_results['margin_corr'])
            results['confidence_entropy_corr'].append(rnn_results['entropy_corr'])
            
            # Train and test Feedforward
            ff_model = FeedforwardBaseline(input_size=3, hidden_size=64, output_size=2)
            ff_model = train_model(ff_model, train_loader, num_epochs=15, lr=1e-3)
            
            ff_results = evaluate_model_comprehensive(ff_model, test_loader)
            
            results['seed'].append(seed)
            results['noise_level'].append(noise)
            results['model_type'].append('Feedforward')
            results['accuracy'].append(ff_results['accuracy'])
            results['confidence_margin_corr'].append(ff_results['margin_corr'])
            results['confidence_entropy_corr'].append(ff_results['entropy_corr'])
    
    return pd.DataFrame(results)


def evaluate_model_comprehensive(model, dataloader):
    """
    Comprehensive evaluation including accuracy and confidence correlations.
    """
    model.eval()
    results = get_hidden_activations(model, dataloader)
    
    # Calculate accuracy
    accuracy = (results['predictions'] == results['labels']).mean()
    
    # Calculate confidence correlations
    correct = (results['predictions'] == results['labels']).astype(float)
    margin_corr, _ = stats.pearsonr(results['confidence_margin'], correct)
    entropy_corr, _ = stats.pearsonr(-results['confidence_entropy'], correct)
    
    return {
        'accuracy': accuracy,
        'margin_corr': margin_corr,
        'entropy_corr': entropy_corr
    }


def plot_validation_results(df):
    """
    Plot statistical validation results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy by noise level
    sns.boxplot(data=df, x='noise_level', y='accuracy', hue='model_type', ax=axes[0,0])
    axes[0,0].set_title('Accuracy vs Noise Level')
    axes[0,0].set_ylabel('Accuracy')
    
    # Confidence-accuracy correlation (margin)
    sns.boxplot(data=df, x='noise_level', y='confidence_margin_corr', hue='model_type', ax=axes[0,1])
    axes[0,1].set_title('Confidence-Accuracy Correlation (Margin)')
    axes[0,1].set_ylabel('Pearson r')
    
    # Confidence-accuracy correlation (entropy)
    sns.boxplot(data=df, x='noise_level', y='confidence_entropy_corr', hue='model_type', ax=axes[1,0])
    axes[1,0].set_title('Confidence-Accuracy Correlation (Entropy)')
    axes[1,0].set_ylabel('Pearson r')
    
    # Accuracy distribution
    sns.histplot(data=df, x='accuracy', hue='model_type', alpha=0.7, ax=axes[1,1])
    axes[1,1].set_title('Accuracy Distribution')
    axes[1,1].set_xlabel('Accuracy')
    
    plt.tight_layout()
    plt.show()


def statistical_analysis(df):
    """
    Perform statistical tests on validation results.
    """
    print("Statistical Analysis Results:")
    print("=" * 50)
    
    for noise in df['noise_level'].unique():
        print(f"\nNoise Level: {noise}")
        noise_data = df[df['noise_level'] == noise]
        
        rnn_acc = noise_data[noise_data['model_type'] == 'RNN']['accuracy']
        ff_acc = noise_data[noise_data['model_type'] == 'Feedforward']['accuracy']
        
        # T-test for accuracy difference
        t_stat, p_val = stats.ttest_ind(rnn_acc, ff_acc)
        
        print(f"  RNN Accuracy: {rnn_acc.mean():.3f} ± {rnn_acc.std():.3f}")
        print(f"  FF Accuracy:  {ff_acc.mean():.3f} ± {ff_acc.std():.3f}")
        print(f"  T-test: t={t_stat:.3f}, p={p_val:.3f}")
        
        if p_val < 0.05:
            better = "RNN" if rnn_acc.mean() > ff_acc.mean() else "Feedforward"
            print(f"  → {better} significantly better (p < 0.05)")
        else:
            print(f"  → No significant difference (p ≥ 0.05)")
    
    # Overall model comparison
    print(f"\nOverall Performance:")
    rnn_overall = df[df['model_type'] == 'RNN']['accuracy']
    ff_overall = df[df['model_type'] == 'Feedforward']['accuracy']
    
    t_stat, p_val = stats.ttest_ind(rnn_overall, ff_overall)
    print(f"  RNN Overall: {rnn_overall.mean():.3f} ± {rnn_overall.std():.3f}")
    print(f"  FF Overall:  {ff_overall.mean():.3f} ± {ff_overall.std():.3f}")
    print(f"  Overall T-test: t={t_stat:.3f}, p={p_val:.3f}")


def noise_sensitivity_analysis():
    """
    Analyze how models degrade with increasing noise.
    """
    print("\nNoise Sensitivity Analysis:")
    print("=" * 30)
    
    noise_range = np.linspace(0.05, 0.8, 10)
    rnn_accs = []
    ff_accs = []
    
    for noise in noise_range:
        # Test data
        test_trials = generate_trials(n_trials=200, noise_std=noise, seq_len=5, seed=42)
        test_loader = prepare_dataloader(test_trials, use_sequences=True)
        
        # Quick training for this noise level
        train_trials = generate_trials(n_trials=500, noise_std=noise, seq_len=5, seed=123)
        train_loader = prepare_dataloader(train_trials, use_sequences=True)
        
        # RNN
        rnn_model = ContextRNN(input_size=3, hidden_size=32, output_size=2, rnn_type="gru")
        rnn_model = train_model(rnn_model, train_loader, num_epochs=10, lr=1e-3)
        rnn_acc = evaluate_model_accuracy(rnn_model, test_loader)
        rnn_accs.append(rnn_acc)
        
        # Feedforward
        ff_model = FeedforwardBaseline(input_size=3, hidden_size=32, output_size=2)
        ff_model = train_model(ff_model, train_loader, num_epochs=10, lr=1e-3)
        ff_acc = evaluate_model_accuracy(ff_model, test_loader)
        ff_accs.append(ff_acc)
        
        print(f"  σ={noise:.2f}: RNN={rnn_acc:.3f}, FF={ff_acc:.3f}")
    
    # Plot noise sensitivity
    plt.figure(figsize=(10, 6))
    plt.plot(noise_range, rnn_accs, 'o-', label='RNN', linewidth=2, markersize=6)
    plt.plot(noise_range, ff_accs, 's-', label='Feedforward', linewidth=2, markersize=6)
    plt.xlabel('Noise Level (σ)')
    plt.ylabel('Accuracy')
    plt.title('Model Robustness to Input Noise')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def evaluate_model_accuracy(model, dataloader):
    """
    Simple accuracy evaluation.
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


if __name__ == "__main__":
    # Run multi-seed validation
    print("Starting multi-seed validation...")
    results_df = run_multi_seed_validation(n_seeds=3, noise_levels=[0.1, 0.3, 0.5])
    
    # Save results
    results_df.to_csv('validation_results.csv', index=False)
    print("Results saved to validation_results.csv")
    
    # Plot results
    plot_validation_results(results_df)
    
    # Statistical analysis
    statistical_analysis(results_df)
    
    # Noise sensitivity analysis
    noise_sensitivity_analysis()