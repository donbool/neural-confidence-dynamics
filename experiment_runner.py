# experiment_runner.py

import os
import json
import datetime
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import ContextRNN
from baseline import FeedforwardBaseline, compare_models, analyze_context_sensitivity
from generate_data import generate_trials
from train import prepare_dataloader, train_model, save_model
from analyze import get_hidden_activations, plot_pca, analyze_confidence_accuracy_correlation
from dynamics import extract_hidden_trajectories, plot_2D_dynamics
from fixed_points import find_fixed_point
from validation import run_multi_seed_validation, plot_validation_results, statistical_analysis, noise_sensitivity_analysis


class ExperimentRunner:
    def __init__(self, config_path=None):
        """
        Comprehensive experiment runner for neural confidence dynamics project.
        """
        self.config = self.load_config(config_path) if config_path else self.default_config()
        self.results_dir = Path(self.config['results_dir'])
        self.results_dir.mkdir(exist_ok=True)
        
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.results_dir / f"experiment_{self.timestamp}"
        self.experiment_dir.mkdir(exist_ok=True)
        
        print(f"Experiment results will be saved to: {self.experiment_dir}")
    
    def default_config(self):
        """Default experiment configuration."""
        return {
            'results_dir': 'results',
            'model': {
                'hidden_size': 64,
                'rnn_type': 'gru'
            },
            'training': {
                'n_trials': 1000,
                'epochs': 20,
                'lr': 1e-3,
                'batch_size': 32,
                'seq_len': 5
            },
            'experiments': {
                'noise_levels': [0.1, 0.2, 0.3, 0.4, 0.5],
                'validation_seeds': 3,
                'context_switching': True,
                'structured_noise': True
            }
        }
    
    def load_config(self, config_path):
        """Load experiment configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def save_config(self):
        """Save current configuration."""
        config_path = self.experiment_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def run_basic_training(self):
        """Run basic training experiment."""
        print("\\n" + "="*50)
        print("EXPERIMENT 1: Basic Training")
        print("="*50)
        
        # Generate training data
        trials = generate_trials(
            n_trials=self.config['training']['n_trials'],
            noise_std=0.25,
            structured_noise=self.config['experiments']['structured_noise'],
            seq_len=self.config['training']['seq_len'],
            context_switch=self.config['experiments']['context_switching'],
            seed=42
        )
        
        dataloader = prepare_dataloader(trials, use_sequences=True, 
                                      batch_size=self.config['training']['batch_size'])
        
        # Train model
        model = ContextRNN(
            input_size=3, 
            hidden_size=self.config['model']['hidden_size'],
            output_size=2, 
            rnn_type=self.config['model']['rnn_type']
        )
        
        trained_model = train_model(
            model, dataloader, 
            num_epochs=self.config['training']['epochs'],
            lr=self.config['training']['lr']
        )
        
        # Save model
        model_path = self.experiment_dir / 'trained_model.pth'
        save_model(trained_model, str(model_path))
        
        return trained_model
    
    def run_confidence_analysis(self, model):
        """Run confidence emergence analysis."""
        print("\\n" + "="*50)
        print("EXPERIMENT 2: Confidence Analysis")
        print("="*50)
        
        # Generate test data
        test_trials = generate_trials(
            n_trials=500, noise_std=0.25, seq_len=5, 
            context_switch=True, seed=123
        )
        test_loader = prepare_dataloader(test_trials, use_sequences=True)
        
        # Extract hidden activations and analyze confidence
        results = get_hidden_activations(model, test_loader)
        
        # Plot PCA
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plot_pca(results['hidden'], results['confidence_margin'], 'margin')
        plt.savefig(self.experiment_dir / 'pca_margin_confidence.png', dpi=300, bbox_inches='tight')
        
        plt.subplot(1, 3, 2)
        plot_pca(results['hidden'], results['confidence_entropy'], 'entropy')
        plt.savefig(self.experiment_dir / 'pca_entropy_confidence.png', dpi=300, bbox_inches='tight')
        
        plt.subplot(1, 3, 3)
        # Analyze correlations
        correlations = analyze_confidence_accuracy_correlation(results)
        plt.savefig(self.experiment_dir / 'confidence_accuracy_correlation.png', dpi=300, bbox_inches='tight')
        
        # Save results
        np.savez(self.experiment_dir / 'confidence_results.npz', 
                 hidden=results['hidden'],
                 labels=results['labels'],
                 predictions=results['predictions'],
                 confidence_margin=results['confidence_margin'],
                 confidence_entropy=results['confidence_entropy'],
                 correlations=correlations)
        
        return results, correlations
    
    def run_baseline_comparison(self):
        """Run baseline comparison experiment."""
        print("\\n" + "="*50)
        print("EXPERIMENT 3: Baseline Comparison")
        print("="*50)
        
        # Temporarily change working directory for baseline comparison
        original_model_path = "trained_model.pth"
        if (self.experiment_dir / 'trained_model.pth').exists():
            # Copy model to current directory for baseline.py
            import shutil
            shutil.copy(self.experiment_dir / 'trained_model.pth', original_model_path)
        
        try:
            comparison_results = compare_models()
            
            plt.savefig(self.experiment_dir / 'baseline_comparison.png', dpi=300, bbox_inches='tight')
            
            # Save comparison results
            with open(self.experiment_dir / 'comparison_results.json', 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            # Context sensitivity analysis
            analyze_context_sensitivity()
            
        finally:
            # Clean up
            if os.path.exists(original_model_path):
                os.remove(original_model_path)
        
        return comparison_results
    
    def run_dynamics_analysis(self, model):
        """Run dynamics analysis experiment."""
        print("\\n" + "="*50)
        print("EXPERIMENT 4: Dynamics Analysis")
        print("="*50)
        
        # Generate data for dynamics analysis
        dynamics_trials = generate_trials(n_trials=300, noise_std=0.2, seq_len=5, seed=456)
        dynamics_loader = prepare_dataloader(dynamics_trials, use_sequences=True)
        
        # Extract hidden trajectories
        hidden = extract_hidden_trajectories(model, dynamics_loader)
        
        # Plot dynamics
        plt.figure(figsize=(10, 8))
        plot_2D_dynamics(hidden, "RNN Hidden Dynamics (PCA)")
        plt.savefig(self.experiment_dir / 'hidden_dynamics.png', dpi=300, bbox_inches='tight')
        
        # Find fixed points (simplified)
        try:
            print("Searching for fixed points...")
            # Example input contexts
            context_inputs = [
                np.array([0.5, 1.0, 0.0]),  # Positive stimulus, context A
                np.array([0.5, 0.0, 1.0]),  # Positive stimulus, context B
                np.array([-0.5, 1.0, 0.0]), # Negative stimulus, context A
                np.array([-0.5, 0.0, 1.0])  # Negative stimulus, context B
            ]
            
            fixed_points = []
            for i, input_vec in enumerate(context_inputs):
                h0 = np.random.randn(self.config['model']['hidden_size'])
                fp = find_fixed_point(model, input_vec, h0)
                fixed_points.append(fp)
                print(f"Fixed point {i+1}: {np.linalg.norm(fp):.3f} (L2 norm)")
            
            # Save fixed points
            np.savez(self.experiment_dir / 'fixed_points.npz', 
                     fixed_points=np.array(fixed_points),
                     context_inputs=np.array(context_inputs))
            
        except Exception as e:
            print(f"Fixed point analysis failed: {e}")
        
        return hidden
    
    def run_validation_study(self):
        """Run statistical validation study."""
        print("\\n" + "="*50)
        print("EXPERIMENT 5: Statistical Validation")
        print("="*50)
        
        # Multi-seed validation
        validation_results = run_multi_seed_validation(
            n_seeds=self.config['experiments']['validation_seeds'],
            noise_levels=self.config['experiments']['noise_levels']
        )
        
        # Save validation results
        validation_results.to_csv(self.experiment_dir / 'validation_results.csv', index=False)
        
        # Plot validation results
        plot_validation_results(validation_results)
        plt.savefig(self.experiment_dir / 'validation_plots.png', dpi=300, bbox_inches='tight')
        
        # Statistical analysis
        print("\\nPerforming statistical analysis...")
        statistical_analysis(validation_results)
        
        # Noise sensitivity
        print("\\nAnalyzing noise sensitivity...")
        noise_sensitivity_analysis()
        plt.savefig(self.experiment_dir / 'noise_sensitivity.png', dpi=300, bbox_inches='tight')
        
        return validation_results
    
    def generate_report(self, results_summary):
        """Generate experiment report."""
        report_path = self.experiment_dir / 'experiment_report.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# Neural Confidence Dynamics Experiment Report\\n\\n")
            f.write(f"**Timestamp:** {self.timestamp}\\n\\n")
            f.write(f"**Configuration:**\\n")
            f.write(f"```json\\n{json.dumps(self.config, indent=2)}\\n```\\n\\n")
            
            f.write(f"## Experiments Completed\\n\\n")
            for i, (exp_name, status) in enumerate(results_summary.items(), 1):
                f.write(f"{i}. **{exp_name}**: {status}\\n")
            
            f.write(f"\\n## Files Generated\\n\\n")
            for file_path in self.experiment_dir.glob('*'):
                if file_path.is_file() and file_path.name != 'experiment_report.md':
                    f.write(f"- `{file_path.name}`\\n")
            
            f.write(f"\\n## Next Steps\\n\\n")
            f.write(f"1. Examine confidence-accuracy correlations in `confidence_results.npz`\\n")
            f.write(f"2. Compare model performance in `validation_results.csv`\\n")
            f.write(f"3. Analyze hidden dynamics in visualization files\\n")
            f.write(f"4. Review fixed point analysis if available\\n")
        
        print(f"\\nExperiment report saved to: {report_path}")
    
    def run_full_experiment(self):
        """Run complete experimental pipeline."""
        print("Starting Neural Confidence Dynamics Experiment Suite")
        print(f"Results directory: {self.experiment_dir}")
        
        # Save configuration
        self.save_config()
        
        results_summary = {}
        
        try:
            # Experiment 1: Basic Training
            model = self.run_basic_training()
            results_summary['Basic Training'] = 'Completed'
            
            # Experiment 2: Confidence Analysis
            confidence_results, correlations = self.run_confidence_analysis(model)
            results_summary['Confidence Analysis'] = 'Completed'
            
            # Experiment 3: Baseline Comparison
            comparison_results = self.run_baseline_comparison()
            results_summary['Baseline Comparison'] = 'Completed'
            
            # Experiment 4: Dynamics Analysis
            hidden_dynamics = self.run_dynamics_analysis(model)
            results_summary['Dynamics Analysis'] = 'Completed'
            
            # Experiment 5: Statistical Validation
            validation_results = self.run_validation_study()
            results_summary['Statistical Validation'] = 'Completed'
            
        except Exception as e:
            print(f"Experiment failed: {e}")
            results_summary[f'ERROR'] = str(e)
        
        # Generate report
        self.generate_report(results_summary)
        
        print("\\n" + "="*50)
        print("EXPERIMENT SUITE COMPLETED")
        print("="*50)
        print(f"Results saved to: {self.experiment_dir}")
        
        return results_summary


def main():
    parser = argparse.ArgumentParser(description='Run Neural Confidence Dynamics Experiments')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--experiment', type=str, choices=['basic', 'confidence', 'baseline', 'dynamics', 'validation', 'full'], 
                       default='full', help='Which experiment to run')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(config_path=args.config)
    
    if args.experiment == 'full':
        runner.run_full_experiment()
    elif args.experiment == 'basic':
        runner.run_basic_training()
    elif args.experiment == 'confidence':
        model = ContextRNN(input_size=3, hidden_size=64, output_size=2, rnn_type="gru")
        model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))
        runner.run_confidence_analysis(model)
    elif args.experiment == 'baseline':
        runner.run_baseline_comparison()
    elif args.experiment == 'dynamics':
        model = ContextRNN(input_size=3, hidden_size=64, output_size=2, rnn_type="gru")
        model.load_state_dict(torch.load("trained_model.pth", map_location="cpu"))
        runner.run_dynamics_analysis(model)
    elif args.experiment == 'validation':
        runner.run_validation_study()


if __name__ == "__main__":
    main()