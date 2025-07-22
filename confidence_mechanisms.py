# confidence_mechanisms.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, pearsonr
from scipy.signal import find_peaks
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from spiking_network import LeakyIntegrateFireNetwork
from drift_diffusion_model import DriftDiffusionModel


@dataclass
class ConfidenceMetrics:
    """Container for various confidence measures."""
    balance_of_evidence: float
    decision_time: float
    neural_variance: float
    population_synchrony: float
    choice_predictive_activity: float
    post_decision_evidence: float
    decision_boundary_distance: float


class NeuralConfidenceAnalyzer:
    """
    Analyzes confidence-related signals from neural network activity.
    
    Implements multiple theories of confidence:
    1. Balance of evidence at decision time
    2. Neural variability and population synchrony
    3. Choice predictive activity
    4. Post-decision evidence accumulation
    5. Distance from decision boundary
    """
    
    def __init__(self):
        pass
    
    def extract_confidence_signals(self, network_results: Dict, 
                                 decision_vars: Dict) -> ConfidenceMetrics:
        """
        Extract multiple confidence-related signals from network activity.
        
        Args:
            network_results: Results from spiking network simulation
            decision_vars: Decision variables from network
            
        Returns:
            ConfidenceMetrics object with all confidence measures
        """
        # 1. Balance of evidence
        balance_conf = self._balance_of_evidence_confidence(
            decision_vars['decision_variable']
        )
        
        # 2. Decision time confidence
        time_conf = self._decision_time_confidence(
            decision_vars['reaction_time']
        )
        
        # 3. Neural variability
        variance_conf = self._neural_variability_confidence(
            network_results['population_rates']
        )
        
        # 4. Population synchrony
        synchrony_conf = self._population_synchrony_confidence(
            network_results['spike_trains']
        )
        
        # 5. Choice predictive activity
        cpa_conf = self._choice_predictive_activity(
            network_results['population_rates'],
            decision_vars['choice']
        )
        
        # 6. Post-decision evidence
        post_dec_conf = self._post_decision_evidence_confidence(
            decision_vars['decision_variable'],
            decision_vars['reaction_time']
        )
        
        # 7. Decision boundary distance
        boundary_conf = self._decision_boundary_distance(
            decision_vars['decision_variable'],
            decision_vars['choice']
        )
        
        return ConfidenceMetrics(
            balance_of_evidence=balance_conf,
            decision_time=time_conf,
            neural_variance=variance_conf,
            population_synchrony=synchrony_conf,
            choice_predictive_activity=cpa_conf,
            post_decision_evidence=post_dec_conf,
            decision_boundary_distance=boundary_conf
        )
    
    def _balance_of_evidence_confidence(self, decision_variable: np.ndarray) -> float:
        """
        Confidence based on final decision variable magnitude.
        Higher absolute values = higher confidence.
        """
        # Use average of final 100ms of decision variable
        final_window = decision_variable[-100:]  # Assuming 1ms timesteps
        final_evidence = np.mean(np.abs(final_window))
        
        # Normalize to [0, 1] range using sigmoid
        confidence = 1 / (1 + np.exp(-2 * final_evidence))
        return confidence
    
    def _decision_time_confidence(self, reaction_time: float, 
                                max_rt: float = 2.0) -> float:
        """
        Confidence inversely related to reaction time.
        Fast decisions = high confidence.
        """
        normalized_rt = min(reaction_time / max_rt, 1.0)
        confidence = 1 - normalized_rt
        return max(confidence, 0.0)
    
    def _neural_variability_confidence(self, population_rates: Dict) -> float:
        """
        Confidence based on trial-to-trial neural variability.
        Low variability = high confidence.
        """
        # Calculate variance across decision populations
        left_rates = population_rates['left_decision']
        right_rates = population_rates['right_decision']
        
        # Combined variance in final decision period
        decision_period = slice(-200, None)  # Last 200ms
        left_var = np.var(left_rates[decision_period])
        right_var = np.var(right_rates[decision_period])
        
        total_variance = left_var + right_var
        
        # Convert to confidence (inverse relationship)
        confidence = 1 / (1 + total_variance)
        return confidence
    
    def _population_synchrony_confidence(self, spike_trains: List) -> float:
        """
        Confidence based on population synchrony during decision period.
        Higher synchrony = higher confidence.
        """
        if len(spike_trains) < 100:
            return 0.5  # Not enough data
        
        # Focus on decision period (last 200 timesteps)
        decision_period_spikes = spike_trains[-200:]
        
        # Count population spike counts per timestep
        spike_counts = [len(spikes) for spikes in decision_period_spikes]
        
        if len(spike_counts) == 0 or max(spike_counts) == 0:
            return 0.5
        
        # Synchrony as coefficient of variation of spike counts
        mean_count = np.mean(spike_counts)
        std_count = np.std(spike_counts)
        
        if mean_count > 0:
            cv = std_count / mean_count
            # Higher CV = lower synchrony = lower confidence
            confidence = 1 / (1 + cv)
        else:
            confidence = 0.5
        
        return confidence
    
    def _choice_predictive_activity(self, population_rates: Dict, 
                                  choice: int) -> float:
        """
        Confidence based on how well early activity predicts final choice.
        Stronger early bias = higher confidence.
        """
        left_rates = population_rates['left_decision']
        right_rates = population_rates['right_decision']
        
        # Compare early activity (first 500ms) with final choice
        early_period = slice(0, 500)  # First 500ms
        
        early_left = np.mean(left_rates[early_period])
        early_right = np.mean(right_rates[early_period])
        
        early_bias = early_left - early_right
        
        # Check consistency with final choice
        if choice == 0:  # Left choice
            consistency = -early_bias  # Want negative bias for left
        else:  # Right choice
            consistency = early_bias   # Want positive bias for right
        
        # Convert to confidence
        confidence = 1 / (1 + np.exp(-consistency))
        return confidence
    
    def _post_decision_evidence_confidence(self, decision_variable: np.ndarray,
                                         reaction_time: float) -> float:
        """
        Confidence based on evidence consistency after decision commitment.
        """
        # Find decision time index (assuming 1ms timesteps)
        rt_idx = int(reaction_time * 1000)
        
        if rt_idx >= len(decision_variable) - 50:
            return 0.5  # Not enough post-decision data
        
        # Post-decision period (50ms after decision)
        post_decision_evidence = decision_variable[rt_idx:rt_idx + 50]
        
        # Check consistency of post-decision evidence with choice
        choice_direction = np.sign(decision_variable[rt_idx])
        
        if choice_direction == 0:
            return 0.5  # Ambiguous choice
        
        # Proportion of post-decision evidence supporting the choice
        consistent_evidence = np.sum(np.sign(post_decision_evidence) == choice_direction)
        total_evidence = len(post_decision_evidence)
        
        consistency = consistent_evidence / total_evidence
        return consistency
    
    def _decision_boundary_distance(self, decision_variable: np.ndarray,
                                  choice: int, threshold: float = 5.0) -> float:
        """
        Confidence based on distance from decision boundary at choice time.
        Further from boundary = higher confidence.
        """
        # Find when decision variable first crossed threshold
        abs_dv = np.abs(decision_variable)
        threshold_crossings = np.where(abs_dv > threshold)[0]
        
        if len(threshold_crossings) == 0:
            # No clear threshold crossing
            final_distance = np.abs(decision_variable[-1])
        else:
            # Distance at first threshold crossing
            crossing_idx = threshold_crossings[0]
            final_distance = abs_dv[crossing_idx]
        
        # Normalize to confidence
        confidence = min(final_distance / (2 * threshold), 1.0)
        return confidence


class MetacognitionAnalyzer:
    """
    Analyzes metacognitive accuracy - how well confidence predicts performance.
    """
    
    @staticmethod
    def calculate_type2_sensitivity(confidences: np.ndarray, 
                                  accuracies: np.ndarray) -> Dict:
        """
        Calculate Type 2 sensitivity (meta-d') for confidence judgments.
        
        Args:
            confidences: Confidence ratings for each trial
            accuracies: Accuracy (0/1) for each trial
            
        Returns:
            Dictionary with metacognitive measures
        """
        from sklearn.metrics import roc_auc_score
        
        # Separate correct and incorrect trials
        correct_trials = accuracies == 1
        incorrect_trials = accuracies == 0
        
        if np.sum(correct_trials) == 0 or np.sum(incorrect_trials) == 0:
            return {'auroc': 0.5, 'meta_sensitivity': 0.0}
        
        # Calculate AUROC for confidence discriminating correct/incorrect
        auroc = roc_auc_score(accuracies, confidences)
        
        # Calculate correlation between confidence and accuracy
        r_conf_acc, p_value = pearsonr(confidences, accuracies)
        
        # Meta-sensitivity (simplified)
        meta_sensitivity = 2 * (auroc - 0.5)  # Convert to d' scale
        
        return {
            'auroc': auroc,
            'meta_sensitivity': meta_sensitivity,
            'confidence_accuracy_correlation': r_conf_acc,
            'p_value': p_value
        }
    
    @staticmethod
    def analyze_confidence_resolution(confidences: np.ndarray,
                                    accuracies: np.ndarray,
                                    n_bins: int = 5) -> Dict:
        """
        Analyze confidence resolution - accuracy within confidence bins.
        """
        # Create confidence bins
        conf_bins = np.linspace(np.min(confidences), np.max(confidences), n_bins + 1)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            # Find trials in this confidence bin
            in_bin = (confidences >= conf_bins[i]) & (confidences < conf_bins[i + 1])
            
            if i == n_bins - 1:  # Include upper bound for last bin
                in_bin = (confidences >= conf_bins[i]) & (confidences <= conf_bins[i + 1])
            
            if np.sum(in_bin) > 0:
                bin_acc = np.mean(accuracies[in_bin])
                bin_conf = np.mean(confidences[in_bin])
                bin_count = np.sum(in_bin)
            else:
                bin_acc = np.nan
                bin_conf = np.nan
                bin_count = 0
            
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_count)
        
        return {
            'bin_confidences': np.array(bin_confidences),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_counts': np.array(bin_counts)
        }
    
    @staticmethod
    def plot_confidence_analysis(confidences: np.ndarray, accuracies: np.ndarray,
                               reaction_times: np.ndarray = None):
        """
        Create comprehensive confidence analysis plots.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Confidence vs Accuracy scatter
        axes[0, 0].scatter(confidences, accuracies, alpha=0.6)
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Confidence vs Accuracy')
        
        # Add correlation
        r, p = pearsonr(confidences, accuracies)
        axes[0, 0].text(0.05, 0.95, f'r = {r:.3f}, p = {p:.3f}', 
                       transform=axes[0, 0].transAxes, verticalalignment='top')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Confidence resolution curve
        resolution = MetacognitionAnalyzer.analyze_confidence_resolution(confidences, accuracies)
        
        valid_bins = ~np.isnan(resolution['bin_accuracies'])
        if np.any(valid_bins):
            axes[0, 1].plot(resolution['bin_confidences'][valid_bins], 
                          resolution['bin_accuracies'][valid_bins], 'o-', 
                          linewidth=2, markersize=8)
            axes[0, 1].plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='Perfect calibration')
        
        axes[0, 1].set_xlabel('Mean Confidence')
        axes[0, 1].set_ylabel('Mean Accuracy')
        axes[0, 1].set_title('Confidence Resolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Confidence distribution
        axes[1, 0].hist(confidences, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Confidence vs RT (if available)
        if reaction_times is not None:
            axes[1, 1].scatter(confidences, reaction_times, alpha=0.6, color='red')
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Reaction Time (s)')
            axes[1, 1].set_title('Confidence vs Reaction Time')
            
            # Add correlation
            r_rt, p_rt = pearsonr(confidences, reaction_times)
            axes[1, 1].text(0.05, 0.95, f'r = {r_rt:.3f}, p = {p_rt:.3f}', 
                           transform=axes[1, 1].transAxes, verticalalignment='top')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Alternative: confidence by accuracy
            correct_conf = confidences[accuracies == 1]
            incorrect_conf = confidences[accuracies == 0]
            
            axes[1, 1].hist([correct_conf, incorrect_conf], bins=15, alpha=0.7, 
                           label=['Correct', 'Incorrect'], color=['green', 'red'])
            axes[1, 1].set_xlabel('Confidence')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Confidence by Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Example usage and testing
if __name__ == "__main__":
    print("Testing confidence mechanisms...")
    
    # Create mock network results for testing
    n_steps = 1000
    mock_rates = {
        'left_decision': 10 + 2 * np.sin(np.linspace(0, 4*np.pi, n_steps)) + np.random.normal(0, 1, n_steps),
        'right_decision': 12 + 3 * np.cos(np.linspace(0, 4*np.pi, n_steps)) + np.random.normal(0, 1, n_steps),
        'confidence': 8 + np.random.normal(0, 0.5, n_steps)
    }
    
    mock_spikes = [np.random.choice(100, size=np.random.poisson(5)) for _ in range(n_steps)]
    
    mock_network_results = {
        'population_rates': mock_rates,
        'spike_trains': mock_spikes
    }
    
    mock_decision_vars = {
        'decision_variable': mock_rates['left_decision'] - mock_rates['right_decision'],
        'choice': 1,
        'reaction_time': 0.8
    }
    
    # Test confidence analyzer
    analyzer = NeuralConfidenceAnalyzer()
    confidence_metrics = analyzer.extract_confidence_signals(
        mock_network_results, mock_decision_vars
    )
    
    print(f"Confidence metrics:")
    print(f"  Balance of evidence: {confidence_metrics.balance_of_evidence:.3f}")
    print(f"  Decision time: {confidence_metrics.decision_time:.3f}")
    print(f"  Neural variance: {confidence_metrics.neural_variance:.3f}")
    print(f"  Population synchrony: {confidence_metrics.population_synchrony:.3f}")
    print(f"  Choice predictive activity: {confidence_metrics.choice_predictive_activity:.3f}")
    print(f"  Post-decision evidence: {confidence_metrics.post_decision_evidence:.3f}")
    print(f"  Boundary distance: {confidence_metrics.decision_boundary_distance:.3f}")
    
    # Test metacognition analysis with simulated data
    n_trials = 200
    simulated_confidences = np.random.beta(2, 2, n_trials)  # U-shaped distribution
    simulated_accuracies = (simulated_confidences + 0.2 * np.random.normal(0, 1, n_trials) > 0.6).astype(int)
    simulated_rts = 1.5 - simulated_confidences + 0.2 * np.random.normal(0, 1, n_trials)
    
    # Analyze metacognition
    meta_results = MetacognitionAnalyzer.calculate_type2_sensitivity(
        simulated_confidences, simulated_accuracies
    )
    
    print(f"\\nMetacognition analysis:")
    print(f"  AUROC: {meta_results['auroc']:.3f}")
    print(f"  Meta-sensitivity: {meta_results['meta_sensitivity']:.3f}")
    print(f"  Confidence-accuracy correlation: {meta_results['confidence_accuracy_correlation']:.3f}")
    
    # Plot confidence analysis
    MetacognitionAnalyzer.plot_confidence_analysis(
        simulated_confidences, simulated_accuracies, simulated_rts
    )
    plt.show()
    
    print("✓ Confidence mechanisms implementation complete")