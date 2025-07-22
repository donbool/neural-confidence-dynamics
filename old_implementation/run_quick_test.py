# run_quick_test.py

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent empty windows

import matplotlib.pyplot as plt
from experiment_runner import ExperimentRunner


def run_quick_test():
    """
    Run a quick test with minimal parameters to check if everything works.
    """
    # Create a quick test configuration
    quick_config = {
        'results_dir': 'test_results',
        'model': {
            'hidden_size': 32,
            'rnn_type': 'gru'
        },
        'training': {
            'n_trials': 200,
            'epochs': 5,
            'lr': 1e-3,
            'batch_size': 16,
            'seq_len': 3
        },
        'experiments': {
            'noise_levels': [0.2, 0.4],
            'validation_seeds': 2,
            'context_switching': True,
            'structured_noise': True
        }
    }
    
    # Create runner with quick config
    runner = ExperimentRunner()
    runner.config = quick_config
    
    # Create test directory
    from pathlib import Path
    import datetime
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    runner.experiment_dir = Path(quick_config['results_dir']) / f"quick_test_{timestamp}"
    runner.experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running quick test - results in: {runner.experiment_dir}")
    
    try:
        # Test basic training
        print("Testing basic training...")
        model = runner.run_basic_training()
        print("✓ Basic training completed")
        
        # Test confidence analysis
        print("Testing confidence analysis...")
        results, correlations = runner.run_confidence_analysis(model)
        print("✓ Confidence analysis completed")
        print(f"  Margin correlation: {correlations['margin_corr']:.3f}")
        print(f"  Entropy correlation: {correlations['entropy_corr']:.3f}")
        
        # Test dynamics analysis
        print("Testing dynamics analysis...")
        hidden = runner.run_dynamics_analysis(model)
        print("✓ Dynamics analysis completed")
        
        print(f"\\n✅ Quick test completed successfully!")
        print(f"Results saved to: {runner.experiment_dir}")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_quick_test()