# perceptual_task.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from dataclasses import dataclass
from typing import Tuple, List, Optional
import random


@dataclass
class TrialResult:
    """Single trial result with all relevant measurements."""
    coherence: float
    direction: int  # 0 = left, 1 = right
    context: str  # 'motion_dir' or 'motion_speed'
    stimulus_sequence: np.ndarray  # (time_steps, 2) - momentary evidence
    choice: int
    confidence: int  # 0 = low, 1 = high
    reaction_time: float
    correct: bool


class MotionCoherenceTask:
    """
    Biologically-inspired motion coherence decision task.
    
    Based on random dot kinematogram paradigm (Newsome & Paré, 1988).
    Implements varying coherence levels and context switching between
    motion direction and speed judgments.
    """
    
    def __init__(self, 
                 dt: float = 0.001,  # 1ms time steps
                 max_duration: float = 2.0,  # Maximum trial duration
                 coherence_levels: List[float] = None,
                 noise_std: float = 1.0):
        
        self.dt = dt
        self.max_duration = max_duration
        self.max_steps = int(max_duration / dt)
        self.noise_std = noise_std
        
        if coherence_levels is None:
            # Standard coherence levels from monkey experiments
            self.coherence_levels = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
        else:
            self.coherence_levels = coherence_levels
    
    def generate_stimulus_sequence(self, coherence: float, direction: int, 
                                 duration_steps: int) -> np.ndarray:
        """
        Generate momentary evidence sequence for motion stimulus.
        
        Args:
            coherence: Motion coherence (0-1)
            direction: True direction (0=left, 1=right)  
            duration_steps: Number of time steps
            
        Returns:
            stimulus: (duration_steps, 2) array of momentary evidence
                     [:, 0] = left evidence, [:, 1] = right evidence
        """
        # Base noise for both directions
        noise = np.random.normal(0, self.noise_std, (duration_steps, 2))
        
        # Add coherent signal
        signal_strength = coherence * self.noise_std
        
        # Direction-specific signal
        if direction == 0:  # Left motion
            signal = np.array([-signal_strength, signal_strength])
        else:  # Right motion  
            signal = np.array([signal_strength, -signal_strength])
        
        # Combine noise and signal
        stimulus = noise + signal[np.newaxis, :]
        
        # Ensure non-negative evidence (like firing rates)
        stimulus = np.maximum(stimulus, 0.1)
        
        return stimulus
    
    def generate_speed_context_stimulus(self, coherence: float, speed: int,
                                      duration_steps: int) -> np.ndarray:
        """
        Generate stimulus for speed discrimination context.
        
        Args:
            coherence: Speed coherence (0-1)
            speed: True speed category (0=slow, 1=fast)
            duration_steps: Number of time steps
        
        Returns:
            stimulus: (duration_steps, 2) array - [slow_evidence, fast_evidence]
        """
        # Base noise
        noise = np.random.normal(0, self.noise_std, (duration_steps, 2))
        
        # Speed signal
        signal_strength = coherence * self.noise_std
        
        if speed == 0:  # Slow
            signal = np.array([signal_strength, -signal_strength])
        else:  # Fast
            signal = np.array([-signal_strength, signal_strength])
        
        stimulus = noise + signal[np.newaxis, :]
        stimulus = np.maximum(stimulus, 0.1)
        
        return stimulus
    
    def generate_trial(self, coherence: Optional[float] = None,
                      direction: Optional[int] = None,
                      context: str = 'motion_dir') -> Tuple[np.ndarray, int, str, float]:
        """
        Generate a single trial stimulus.
        
        Returns:
            stimulus: (max_steps, 2) stimulus sequence
            true_direction: Correct answer
            context: Task context
            coherence: Stimulus coherence
        """
        if coherence is None:
            coherence = random.choice(self.coherence_levels)
        
        if direction is None:
            direction = random.choice([0, 1])
        
        if context == 'motion_dir':
            stimulus = self.generate_stimulus_sequence(
                coherence, direction, self.max_steps
            )
        elif context == 'motion_speed':
            # For speed context, map direction to speed
            speed = direction  # 0=slow, 1=fast
            stimulus = self.generate_speed_context_stimulus(
                coherence, speed, self.max_steps
            )
        else:
            raise ValueError(f"Unknown context: {context}")
        
        return stimulus, direction, context, coherence
    
    def create_experiment_session(self, 
                                n_trials: int = 1000,
                                context_switch_prob: float = 0.1,
                                balanced_conditions: bool = True) -> List[Tuple]:
        """
        Create a full experimental session with multiple trials.
        
        Args:
            n_trials: Number of trials
            context_switch_prob: Probability of context switch between trials
            balanced_conditions: Balance coherence and direction conditions
            
        Returns:
            trials: List of (stimulus, direction, context, coherence) tuples
        """
        trials = []
        current_context = 'motion_dir'
        
        # Create balanced conditions if requested
        if balanced_conditions:
            # Create all combinations of coherence x direction
            conditions = []
            for coh in self.coherence_levels:
                for direction in [0, 1]:
                    conditions.append((coh, direction))
            
            # Repeat to reach desired trial count
            n_repeats = n_trials // len(conditions) + 1
            all_conditions = conditions * n_repeats
            random.shuffle(all_conditions)
            all_conditions = all_conditions[:n_trials]
        else:
            all_conditions = [(None, None)] * n_trials
        
        for i, (coherence, direction) in enumerate(all_conditions):
            # Context switching
            if i > 0 and random.random() < context_switch_prob:
                current_context = 'motion_speed' if current_context == 'motion_dir' else 'motion_dir'
            
            stimulus, true_dir, context, coh = self.generate_trial(
                coherence, direction, current_context
            )
            
            trials.append((stimulus, true_dir, context, coh))
        
        return trials


class PsychophysicalAnalysis:
    """
    Analysis tools for psychophysical performance.
    """
    
    @staticmethod
    def fit_psychometric_curve(coherences: np.ndarray, 
                             accuracies: np.ndarray) -> Tuple[float, float]:
        """
        Fit psychometric function: P(correct) = 0.5 + 0.5 * erf(coherence / (sqrt(2) * sigma))
        
        Returns:
            threshold: Coherence for 75% correct (threshold)
            slope: Psychometric slope parameter
        """
        from scipy.optimize import minimize
        
        def psychometric(c, sigma):
            return 0.5 + 0.5 * np.array([norm.cdf(x / sigma) for x in c])
        
        def loss(params):
            sigma = params[0]
            pred = psychometric(coherences, sigma)
            return np.sum((pred - accuracies) ** 2)
        
        result = minimize(loss, [0.1], bounds=[(0.001, 1.0)])
        sigma = result.x[0]
        threshold = sigma * norm.ppf(0.75)  # 75% correct threshold
        
        return threshold, sigma
    
    @staticmethod
    def plot_psychometric_curve(coherences: np.ndarray, accuracies: np.ndarray,
                               reaction_times: np.ndarray = None):
        """
        Plot psychometric curve and optionally chronometric curve.
        """
        fig, axes = plt.subplots(1, 2 if reaction_times is not None else 1, 
                                figsize=(12, 5))
        
        if reaction_times is None:
            axes = [axes]
        
        # Psychometric curve
        axes[0].plot(coherences, accuracies, 'o-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Motion Coherence')
        axes[0].set_ylabel('Proportion Correct')
        axes[0].set_title('Psychometric Function')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0.4, 1.0)
        
        # Fit and plot smooth curve
        threshold, slope = PsychophysicalAnalysis.fit_psychometric_curve(coherences, accuracies)
        coh_smooth = np.linspace(0, coherences.max(), 100)
        acc_smooth = 0.5 + 0.5 * np.array([norm.cdf(c / slope) for c in coh_smooth])
        axes[0].plot(coh_smooth, acc_smooth, '--', alpha=0.7, 
                    label=f'Threshold: {threshold:.3f}')
        axes[0].legend()
        
        # Chronometric curve (RT vs coherence)
        if reaction_times is not None:
            axes[1].plot(coherences, reaction_times, 's-', linewidth=2, markersize=8, color='red')
            axes[1].set_xlabel('Motion Coherence') 
            axes[1].set_ylabel('Reaction Time (s)')
            axes[1].set_title('Chronometric Function')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def analyze_confidence_accuracy_relationship(confidences: np.ndarray,
                                               accuracies: np.ndarray,
                                               coherences: np.ndarray):
        """
        Analyze relationship between confidence and accuracy (metacognition).
        """
        # Type 2 ROC analysis for confidence
        from sklearn.metrics import roc_auc_score
        
        # Calculate confidence for correct vs incorrect trials
        correct_conf = confidences[accuracies == 1]
        incorrect_conf = confidences[accuracies == 0]
        
        # Type 2 sensitivity (meta-d')
        if len(correct_conf) > 0 and len(incorrect_conf) > 0:
            # Create labels for ROC: 1 = correct trial, 0 = incorrect trial
            labels = np.concatenate([np.ones(len(correct_conf)), 
                                   np.zeros(len(incorrect_conf))])
            confs = np.concatenate([correct_conf, incorrect_conf])
            
            meta_sensitivity = roc_auc_score(labels, confs)
        else:
            meta_sensitivity = 0.5
        
        return {
            'meta_sensitivity': meta_sensitivity,
            'mean_conf_correct': np.mean(correct_conf) if len(correct_conf) > 0 else 0,
            'mean_conf_incorrect': np.mean(incorrect_conf) if len(incorrect_conf) > 0 else 0
        }


# Example usage and testing
if __name__ == "__main__":
    # Create task
    task = MotionCoherenceTask()
    
    # Generate sample trials
    print("Generating sample experimental session...")
    trials = task.create_experiment_session(n_trials=100, balanced_conditions=True)
    
    print(f"Generated {len(trials)} trials")
    print(f"Coherence levels: {task.coherence_levels}")
    
    # Analyze coherence distribution
    coherences = [trial[3] for trial in trials]
    contexts = [trial[2] for trial in trials]
    
    print(f"Motion direction trials: {contexts.count('motion_dir')}")
    print(f"Motion speed trials: {contexts.count('motion_speed')}")
    
    # Plot sample stimulus
    sample_stimulus, direction, context, coherence = trials[0]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(sample_stimulus[:500, 0], label='Left Evidence')
    plt.plot(sample_stimulus[:500, 1], label='Right Evidence') 
    plt.xlabel('Time (ms)')
    plt.ylabel('Evidence')
    plt.title(f'Sample Stimulus (Coherence={coherence:.3f}, Direction={direction})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(coherences, bins=10, alpha=0.7, edgecolor='black')
    plt.xlabel('Coherence')
    plt.ylabel('Count')
    plt.title('Coherence Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Perceptual task implementation complete")