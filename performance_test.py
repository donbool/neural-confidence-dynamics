# performance_test.py

import time
import numpy as np
from spiking_network import LeakyIntegrateFireNetwork, NetworkParams
from perceptual_task import MotionCoherenceTask

def test_single_trial_performance():
    """Test how long a single trial takes."""
    print("Testing single trial performance on M1 Mac...")
    
    # Create small network for testing
    params = NetworkParams(
        n_sensory=50,
        n_decision=30,
        n_confidence=15,
        n_inhibitory=10
    )
    
    network = LeakyIntegrateFireNetwork(params)
    task = MotionCoherenceTask()
    
    # Generate short stimulus
    stimulus, _, _, _ = task.generate_trial()
    stimulus = stimulus[:1000]  # 1 second trial
    
    print(f"Network size: {network.n_total} neurons")
    print(f"Stimulus duration: {len(stimulus)} timesteps (1.0s)")
    
    # Time a single simulation
    start_time = time.time()
    results = network.simulate_trial(stimulus)
    end_time = time.time()
    
    trial_time = end_time - start_time
    print(f"Single trial time: {trial_time:.2f} seconds")
    
    return trial_time

def estimate_full_experiment_time():
    """Estimate full experiment runtime."""
    print("\nEstimating full experiment time...")
    
    # Test parameters from experiment_framework.py
    coherence_levels = [0.0, 0.128, 0.256, 0.512]  # 4 levels
    n_trials_per_condition = 25  # 25 trials each
    total_trials = len(coherence_levels) * 2 * n_trials_per_condition  # 200 trials
    
    # Get single trial time
    single_trial_time = test_single_trial_performance()
    
    # Estimate total time
    estimated_total = single_trial_time * total_trials
    estimated_minutes = estimated_total / 60
    
    print(f"\nFull experiment estimate:")
    print(f"  Total trials: {total_trials}")
    print(f"  Time per trial: {single_trial_time:.2f}s")
    print(f"  Total estimated time: {estimated_minutes:.1f} minutes ({estimated_total/3600:.1f} hours)")
    
    if estimated_minutes > 30:
        print(f"  ⚠️  This might be too slow for interactive testing!")
        print(f"  💡 Consider using the quick test version below.")
    else:
        print(f"  ✅ Should be reasonable for M1 Mac")
    
    return estimated_total

if __name__ == "__main__":
    estimate_full_experiment_time()