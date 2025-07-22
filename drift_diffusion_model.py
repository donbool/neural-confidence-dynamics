# drift_diffusion_model.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional


@dataclass 
class DDMParams:
    """Parameters for Drift Diffusion Model."""
    drift: float = 2.0           # Drift rate (evidence strength)
    boundary: float = 1.0        # Decision boundary
    non_decision_time: float = 0.3  # Non-decision time (s)
    noise_std: float = 1.0       # Diffusion coefficient
    starting_point: float = 0.0  # Starting point bias
    dt: float = 0.001           # Time step for simulation


class DriftDiffusionModel:
    """
    Drift Diffusion Model for two-alternative forced choice decisions.
    
    Based on Ratcliff & McKoon (2008) and Shadlen & Newsome (2001).
    Models decision-making as evidence accumulation to threshold.
    
    Key features:
    - Varying drift rates for different coherence levels
    - Confidence based on evidence at decision time
    - Reaction time predictions
    - Optimal decision boundaries
    """
    
    def __init__(self, params: DDMParams = None):
        self.params = params or DDMParams()
    
    def simulate_single_trial(self, coherence: float, true_direction: int,
                            max_time: float = 5.0) -> Dict:
        """
        Simulate single DDM trial.
        
        Args:
            coherence: Motion coherence (0-1)
            true_direction: True direction (0=left, 1=right)
            max_time: Maximum trial duration
            
        Returns:
            trial_result: Dictionary with choice, RT, confidence
        """
        p = self.params
        
        # Drift rate proportional to coherence
        if true_direction == 1:  # Right
            drift = coherence * p.drift
        else:  # Left  
            drift = -coherence * p.drift
        
        # Initialize evidence accumulator
        evidence = p.starting_point
        time_elapsed = 0
        evidence_trace = [evidence]
        
        # Accumulate evidence until boundary is reached
        while abs(evidence) < p.boundary and time_elapsed < max_time:
            # Add drift and noise
            evidence += drift * p.dt + np.random.normal(0, p.noise_std * np.sqrt(p.dt))
            evidence_trace.append(evidence)
            time_elapsed += p.dt
        
        # Decision and reaction time
        if evidence >= p.boundary:
            choice = 1  # Right
            decision_time = time_elapsed
        elif evidence <= -p.boundary:
            choice = 0  # Left
            decision_time = time_elapsed
        else:
            # Max time reached - forced choice based on evidence
            choice = 1 if evidence > 0 else 0
            decision_time = max_time
        
        # Total reaction time includes non-decision time
        reaction_time = decision_time + p.non_decision_time
        
        # Confidence based on evidence at decision time
        final_evidence = evidence_trace[-1]
        confidence = self._evidence_to_confidence(final_evidence)
        
        # Check if correct
        correct = (choice == true_direction)
        
        return {
            'choice': choice,
            'reaction_time': reaction_time,
            'decision_time': decision_time, 
            'confidence': confidence,
            'correct': correct,
            'final_evidence': final_evidence,
            'evidence_trace': np.array(evidence_trace),
            'coherence': coherence,
            'true_direction': true_direction
        }
    
    def _evidence_to_confidence(self, evidence: float) -> float:
        """
        Convert final evidence to confidence rating.
        
        Based on the idea that confidence reflects distance from
        decision boundary at choice time.
        """
        # Confidence as sigmoid of absolute evidence
        confidence = 1 / (1 + np.exp(-2 * abs(evidence)))
        return confidence
    
    def simulate_experiment(self, coherence_levels: List[float],
                          n_trials_per_condition: int = 100) -> List[Dict]:
        """
        Simulate full psychophysical experiment.
        
        Args:
            coherence_levels: List of motion coherence values
            n_trials_per_condition: Trials per coherence x direction
            
        Returns:
            all_trials: List of trial results
        """
        all_trials = []
        
        for coherence in coherence_levels:
            for direction in [0, 1]:  # Left, Right
                for _ in range(n_trials_per_condition):
                    trial = self.simulate_single_trial(coherence, direction)
                    all_trials.append(trial)
        
        return all_trials
    
    def fit_psychometric_data(self, coherences: np.ndarray, 
                            choices: np.ndarray,
                            reaction_times: np.ndarray = None) -> Dict:
        """
        Fit DDM parameters to behavioral data.
        
        Args:
            coherences: Motion coherence for each trial
            choices: Choice for each trial (0=left, 1=right) 
            reaction_times: Reaction times (optional)
            
        Returns:
            fitted_params: Dictionary of fitted parameters
        """
        def psychometric_likelihood(params):
            """Negative log-likelihood for psychometric data."""
            drift_scale, boundary, non_dec = params
            
            log_likelihood = 0
            
            for coh, choice in zip(coherences, choices):
                # Predicted choice probability for this coherence
                p_right = self._choice_probability(coh, drift_scale, boundary)
                
                # Likelihood of observed choice
                if choice == 1:
                    prob = p_right
                else:
                    prob = 1 - p_right
                
                # Add to log likelihood (avoid log(0))
                log_likelihood += np.log(max(prob, 1e-10))
            
            return -log_likelihood  # Negative for minimization
        
        # Initial parameter guess
        init_params = [self.params.drift, self.params.boundary, self.params.non_decision_time]
        bounds = [(0.1, 10), (0.1, 3.0), (0.1, 1.0)]
        
        # Optimize
        result = minimize(psychometric_likelihood, init_params, bounds=bounds)
        
        # Update parameters
        fitted_drift, fitted_boundary, fitted_non_dec = result.x
        
        return {
            'drift_rate': fitted_drift,
            'boundary': fitted_boundary,
            'non_decision_time': fitted_non_dec,
            'log_likelihood': -result.fun,
            'fit_success': result.success
        }
    
    def _choice_probability(self, coherence: float, drift_scale: float, 
                          boundary: float) -> float:
        """
        Analytical choice probability for given coherence.
        
        Based on first passage time analysis of DDM.
        """
        if coherence == 0:
            return 0.5  # No bias at zero coherence
        
        # Drift rate for this coherence
        drift = coherence * drift_scale
        
        # Choice probability (exact solution for DDM)
        if abs(drift) < 1e-10:  # Avoid division by zero
            return 0.5
        
        # Probability of hitting upper boundary first
        z0 = self.params.starting_point
        exp_term = np.exp(-2 * drift * (boundary + z0) / (self.params.noise_std ** 2))
        p_upper = (1 - exp_term) / (1 - np.exp(-4 * drift * boundary / (self.params.noise_std ** 2)))
        
        return p_upper
    
    def predict_reaction_times(self, coherences: np.ndarray,
                             fitted_params: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict mean reaction times for different coherences.
        
        Returns:
            rt_correct: RTs for correct trials
            rt_error: RTs for error trials
        """
        drift_scale = fitted_params['drift_rate']
        boundary = fitted_params['boundary']
        non_dec = fitted_params['non_decision_time']
        
        rt_correct = []
        rt_error = []
        
        for coh in coherences:
            if coh == 0:
                # Special case for zero coherence
                rt_c = rt_e = boundary**2 / (self.params.noise_std**2) + non_dec
            else:
                drift = coh * drift_scale
                
                # Mean first passage times (analytical approximations)
                # These are simplified - full solutions involve infinite series
                
                # Correct trials (upper boundary for positive drift)
                if drift > 0:
                    rt_c = boundary / drift + non_dec
                    rt_e = -boundary / drift + non_dec if drift != 0 else float('inf')
                else:
                    rt_c = -boundary / drift + non_dec  
                    rt_e = boundary / drift + non_dec
                
                # Ensure positive RTs
                rt_c = max(rt_c, non_dec)
                rt_e = max(rt_e, non_dec)
            
            rt_correct.append(rt_c)
            rt_error.append(rt_e)
        
        return np.array(rt_correct), np.array(rt_error)


class ConfidenceModel:
    """
    Models for confidence in perceptual decisions.
    
    Implements different theories of confidence:
    1. Balance of evidence (Vickers, 1979)
    2. Time to decision (Kiani et al., 2014)
    3. Post-decision evidence (Pleskac & Busemeyer, 2010)
    """
    
    @staticmethod
    def balance_of_evidence(evidence_trace: np.ndarray, boundary: float) -> float:
        """
        Confidence based on final evidence relative to boundary.
        """
        final_evidence = evidence_trace[-1]
        confidence = abs(final_evidence) / boundary
        return min(confidence, 1.0)  # Cap at 1.0
    
    @staticmethod  
    def time_to_decision(decision_time: float, max_time: float = 2.0) -> float:
        """
        Confidence inversely related to decision time.
        Fast decisions = high confidence.
        """
        normalized_time = decision_time / max_time
        confidence = 1 - normalized_time
        return max(confidence, 0.0)
    
    @staticmethod
    def decision_variable_variance(evidence_trace: np.ndarray, 
                                 window_size: int = 50) -> float:
        """
        Confidence based on stability of decision variable.
        Low variance = high confidence.
        """
        if len(evidence_trace) < window_size:
            window_size = len(evidence_trace)
        
        # Variance in final window
        final_window = evidence_trace[-window_size:]
        variance = np.var(final_window)
        
        # Convert to confidence (inverse relationship)
        confidence = 1 / (1 + variance)
        return confidence
    
    @staticmethod
    def post_decision_evidence(evidence_trace: np.ndarray, 
                             decision_time_idx: int,
                             post_decision_window: int = 100) -> float:
        """
        Confidence based on evidence after decision commitment.
        """
        if decision_time_idx + post_decision_window >= len(evidence_trace):
            return 0.5  # Not enough post-decision data
        
        post_evidence = evidence_trace[decision_time_idx:decision_time_idx + post_decision_window]
        
        # Consistency of post-decision evidence with choice
        choice_direction = np.sign(evidence_trace[decision_time_idx])
        consistency = np.mean(np.sign(post_evidence) == choice_direction)
        
        return consistency


# Example usage and validation
if __name__ == "__main__":
    # Create DDM
    print("Testing Drift Diffusion Model...")
    ddm = DriftDiffusionModel()
    
    # Test single trial
    trial = ddm.simulate_single_trial(coherence=0.32, true_direction=1)
    print(f"Single trial result:")
    print(f"  Choice: {'Right' if trial['choice'] == 1 else 'Left'}")
    print(f"  RT: {trial['reaction_time']:.3f}s")  
    print(f"  Confidence: {trial['confidence']:.3f}")
    print(f"  Correct: {trial['correct']}")
    
    # Simulate psychophysical experiment
    coherences = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
    print(f"\\nSimulating experiment with coherences: {coherences}")
    
    trials = ddm.simulate_experiment(coherences, n_trials_per_condition=50)
    print(f"Simulated {len(trials)} trials")
    
    # Analyze results
    coherence_vals = [t['coherence'] for t in trials]
    choices = [t['choice'] for t in trials]
    rts = [t['reaction_time'] for t in trials]
    confidences = [t['confidence'] for t in trials]
    correct = [t['correct'] for t in trials]
    
    # Calculate psychometric curve
    unique_coherences = sorted(set(coherence_vals))
    accuracies = []
    mean_rts = []
    mean_confidences = []
    
    for coh in unique_coherences:
        coh_trials = [t for t in trials if t['coherence'] == coh]
        
        # Accuracy
        acc = np.mean([t['correct'] for t in coh_trials])
        accuracies.append(acc)
        
        # Mean RT
        rt = np.mean([t['reaction_time'] for t in coh_trials])
        mean_rts.append(rt)
        
        # Mean confidence
        conf = np.mean([t['confidence'] for t in coh_trials])
        mean_confidences.append(conf)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Psychometric curve
    axes[0, 0].plot(unique_coherences, accuracies, 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Coherence')
    axes[0, 0].set_ylabel('Proportion Correct')
    axes[0, 0].set_title('Psychometric Function (DDM)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0.4, 1.0)
    
    # Chronometric curve
    axes[0, 1].plot(unique_coherences, mean_rts, 's-', linewidth=2, markersize=8, color='red')
    axes[0, 1].set_xlabel('Coherence')
    axes[0, 1].set_ylabel('Reaction Time (s)')
    axes[0, 1].set_title('Chronometric Function (DDM)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confidence curve
    axes[1, 0].plot(unique_coherences, mean_confidences, '^-', linewidth=2, markersize=8, color='green')
    axes[1, 0].set_xlabel('Coherence')
    axes[1, 0].set_ylabel('Mean Confidence')
    axes[1, 0].set_title('Confidence vs Coherence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Example evidence trace
    sample_trial = trials[50]  # Pick a sample trial
    time_axis = np.arange(len(sample_trial['evidence_trace'])) * ddm.params.dt
    axes[1, 1].plot(time_axis, sample_trial['evidence_trace'], 'k-', linewidth=2)
    axes[1, 1].axhline(ddm.params.boundary, color='red', linestyle='--', label='Upper Boundary')
    axes[1, 1].axhline(-ddm.params.boundary, color='red', linestyle='--', label='Lower Boundary')
    axes[1, 1].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Evidence')
    axes[1, 1].set_title(f'Sample Evidence Trace (Coherence={sample_trial["coherence"]:.3f})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Test confidence models
    print("\\nTesting confidence models:")
    sample_evidence = sample_trial['evidence_trace']
    decision_idx = len(sample_evidence) - 100  # Decision made near end
    
    conf_balance = ConfidenceModel.balance_of_evidence(sample_evidence, ddm.params.boundary)
    conf_time = ConfidenceModel.time_to_decision(sample_trial['decision_time'])
    conf_variance = ConfidenceModel.decision_variable_variance(sample_evidence)
    
    print(f"  Balance of evidence: {conf_balance:.3f}")
    print(f"  Time to decision: {conf_time:.3f}")
    print(f"  Decision variance: {conf_variance:.3f}")
    
    print("✓ Drift Diffusion Model implementation complete")