# experiment_framework.py

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import pandas as pd

from perceptual_task import MotionCoherenceTask, PsychophysicalAnalysis, TrialResult
from spiking_network import LeakyIntegrateFireNetwork, NetworkParams
from drift_diffusion_model import DriftDiffusionModel, DDMParams
from confidence_mechanisms import NeuralConfidenceAnalyzer, MetacognitionAnalyzer


@dataclass
class ExperimentConfig:
    """Configuration for neural confidence experiment."""
    # Task parameters
    coherence_levels: List[float] = None
    n_trials_per_condition: int = 50
    max_trial_duration: float = 2.0
    context_switch_prob: float = 0.1
    
    # Network parameters  
    network_params: NetworkParams = None
    
    # DDM parameters
    ddm_params: DDMParams = None
    
    # Analysis parameters
    confidence_threshold: float = 0.6
    rt_bins: List[float] = None
    
    # Experiment metadata
    experiment_name: str = "neural_confidence_dynamics"
    description: str = "Biologically plausible investigation of confidence in perceptual decisions"
    
    def __post_init__(self):
        if self.coherence_levels is None:
            self.coherence_levels = [0.0, 0.032, 0.064, 0.128, 0.256, 0.512]
        
        if self.network_params is None:
            self.network_params = NetworkParams()
        
        if self.ddm_params is None:
            self.ddm_params = DDMParams()
        
        if self.rt_bins is None:
            self.rt_bins = [0.3, 0.5, 0.8, 1.2, 2.0]


@dataclass
class ExperimentResults:
    """Container for all experimental results."""
    # Behavioral data
    behavioral_results: List[Dict]
    spiking_results: List[Dict] 
    ddm_results: List[Dict]
    
    # Analysis results
    psychometric_analysis: Dict
    confidence_analysis: Dict
    metacognition_analysis: Dict
    model_comparison: Dict
    
    # Metadata
    config: ExperimentConfig
    timestamp: str
    duration_minutes: float


class NeuralConfidenceExperiment:
    """
    Complete framework for neural confidence dynamics experiment.
    
    This class orchestrates the entire experimental pipeline:
    1. Generate perceptual stimuli with varying coherence
    2. Simulate spiking neural network responses
    3. Compare with drift-diffusion model predictions
    4. Analyze confidence mechanisms and metacognition
    5. Validate against psychophysical data
    """
    
    def __init__(self, config: ExperimentConfig = None, results_dir: str = "results"):
        self.config = config or ExperimentConfig()
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.task = MotionCoherenceTask(
            coherence_levels=self.config.coherence_levels,
            max_duration=self.config.max_trial_duration
        )
        
        self.spiking_network = LeakyIntegrateFireNetwork(self.config.network_params)
        self.ddm = DriftDiffusionModel(self.config.ddm_params)
        self.confidence_analyzer = NeuralConfidenceAnalyzer()
        
        print(f"Experiment initialized: {self.config.experiment_name}")
        print(f"Results will be saved to: {self.results_dir}")
    
    def run_full_experiment(self) -> ExperimentResults:
        """
        Run the complete experimental pipeline.
        
        Returns:
            ExperimentResults with all data and analyses
        """
        start_time = datetime.datetime.now()
        print(f"Starting experiment: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Phase 1: Generate experimental session
        print("\\n=== Phase 1: Generating Experimental Session ===")
        trials = self.task.create_experiment_session(
            n_trials=len(self.config.coherence_levels) * 2 * self.config.n_trials_per_condition,
            context_switch_prob=self.config.context_switch_prob,
            balanced_conditions=True
        )
        print(f"Generated {len(trials)} trials")
        
        # Phase 2: Simulate neural responses
        print("\\n=== Phase 2: Simulating Neural Responses ===")
        behavioral_results, spiking_results = self._simulate_neural_responses(trials)
        print(f"Completed {len(spiking_results)} neural simulations")
        
        # Phase 3: Simulate DDM responses
        print("\\n=== Phase 3: Simulating DDM Responses ===")  
        ddm_results = self._simulate_ddm_responses(trials)
        print(f"Completed {len(ddm_results)} DDM simulations")
        
        # Phase 4: Analyze behavioral performance
        print("\\n=== Phase 4: Analyzing Behavioral Performance ===")
        psychometric_analysis = self._analyze_psychometric_performance(behavioral_results)
        
        # Phase 5: Analyze confidence mechanisms
        print("\\n=== Phase 5: Analyzing Confidence Mechanisms ===")
        confidence_analysis = self._analyze_confidence_mechanisms(
            behavioral_results, spiking_results
        )
        
        # Phase 6: Analyze metacognition
        print("\\n=== Phase 6: Analyzing Metacognition ===")
        metacognition_analysis = self._analyze_metacognition(behavioral_results)
        
        # Phase 7: Compare models
        print("\\n=== Phase 7: Comparing Models ===")
        model_comparison = self._compare_models(behavioral_results, ddm_results)
        
        # Create results object
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        results = ExperimentResults(
            behavioral_results=behavioral_results,
            spiking_results=spiking_results,
            ddm_results=ddm_results,
            psychometric_analysis=psychometric_analysis,
            confidence_analysis=confidence_analysis,
            metacognition_analysis=metacognition_analysis,
            model_comparison=model_comparison,
            config=self.config,
            timestamp=start_time.isoformat(),
            duration_minutes=duration
        )
        
        # Save results
        self._save_results(results)
        
        print(f"\\n=== Experiment Complete ===")
        print(f"Duration: {duration:.1f} minutes")
        print(f"Results saved to: {self.results_dir}")
        
        return results
    
    def _simulate_neural_responses(self, trials: List) -> Tuple[List[Dict], List[Dict]]:
        """Simulate spiking network responses to all trials."""
        behavioral_results = []
        spiking_results = []
        
        for i, (stimulus, true_direction, context, coherence) in enumerate(trials):
            if i % 100 == 0:
                print(f"  Simulating trial {i+1}/{len(trials)}")
            
            # Run spiking network simulation
            network_results = self.spiking_network.simulate_trial(stimulus)
            decision_vars = self.spiking_network.extract_decision_variables(network_results)
            
            # Extract confidence signals
            confidence_metrics = self.confidence_analyzer.extract_confidence_signals(
                network_results, decision_vars
            )
            
            # Overall confidence (weighted combination of metrics)
            overall_confidence = self._compute_overall_confidence(confidence_metrics)
            
            # Store behavioral result
            behavioral_result = {
                'trial': i,
                'coherence': coherence,
                'true_direction': true_direction,
                'context': context,
                'choice': decision_vars['choice'],
                'reaction_time': decision_vars['reaction_time'],
                'confidence': overall_confidence,
                'correct': (decision_vars['choice'] == true_direction)
            }
            behavioral_results.append(behavioral_result)
            
            # Store detailed spiking result
            spiking_result = {
                'trial': i,
                'network_results': network_results,
                'decision_vars': decision_vars,
                'confidence_metrics': confidence_metrics
            }
            spiking_results.append(spiking_result)
        
        return behavioral_results, spiking_results
    
    def _simulate_ddm_responses(self, trials: List) -> List[Dict]:
        """Simulate DDM responses to all trials."""
        ddm_results = []
        
        for i, (_, true_direction, context, coherence) in enumerate(trials):
            # Run DDM simulation
            ddm_result = self.ddm.simulate_single_trial(coherence, true_direction)
            ddm_result['trial'] = i
            ddm_result['context'] = context
            ddm_results.append(ddm_result)
        
        return ddm_results
    
    def _compute_overall_confidence(self, confidence_metrics) -> float:
        """
        Compute overall confidence from multiple neural mechanisms.
        
        Weighted combination based on theoretical importance.
        """
        weights = {
            'balance_of_evidence': 0.25,
            'decision_time': 0.20,
            'neural_variance': 0.15,
            'population_synchrony': 0.10,
            'choice_predictive_activity': 0.15,
            'post_decision_evidence': 0.10,
            'decision_boundary_distance': 0.05
        }
        
        overall = 0
        for metric, weight in weights.items():
            value = getattr(confidence_metrics, metric)
            overall += weight * value
        
        return overall
    
    def _analyze_psychometric_performance(self, behavioral_results: List[Dict]) -> Dict:
        """Analyze psychometric and chronometric functions."""
        # Convert to arrays
        coherences = np.array([r['coherence'] for r in behavioral_results])
        choices = np.array([r['choice'] for r in behavioral_results])
        rts = np.array([r['reaction_time'] for r in behavioral_results])
        correct = np.array([r['correct'] for r in behavioral_results])
        
        # Group by coherence
        unique_coherences = sorted(set(coherences))
        mean_accuracies = []
        mean_rts = []
        
        for coh in unique_coherences:
            coh_mask = coherences == coh
            mean_acc = np.mean(correct[coh_mask])
            mean_rt = np.mean(rts[coh_mask])
            mean_accuracies.append(mean_acc)
            mean_rts.append(mean_rt)
        
        # Fit psychometric curve
        threshold, slope = PsychophysicalAnalysis.fit_psychometric_curve(
            np.array(unique_coherences), np.array(mean_accuracies)
        )
        
        return {
            'coherences': unique_coherences,
            'accuracies': mean_accuracies,
            'reaction_times': mean_rts,
            'threshold': threshold,
            'slope': slope,
            'overall_accuracy': np.mean(correct),
            'overall_rt': np.mean(rts)
        }
    
    def _analyze_confidence_mechanisms(self, behavioral_results: List[Dict], 
                                     spiking_results: List[Dict]) -> Dict:
        """Analyze confidence mechanisms from neural data."""
        # Extract confidence metrics for each mechanism
        mechanisms = [
            'balance_of_evidence', 'decision_time', 'neural_variance',
            'population_synchrony', 'choice_predictive_activity',
            'post_decision_evidence', 'decision_boundary_distance'
        ]
        
        mechanism_confidences = {mech: [] for mech in mechanisms}
        overall_confidences = []
        accuracies = []
        
        for behav, spike in zip(behavioral_results, spiking_results):
            overall_confidences.append(behav['confidence'])
            accuracies.append(behav['correct'])
            
            for mech in mechanisms:
                value = getattr(spike['confidence_metrics'], mech)
                mechanism_confidences[mech].append(value)
        
        # Analyze each mechanism's relationship with accuracy
        mechanism_analysis = {}
        for mech in mechanisms:
            confs = np.array(mechanism_confidences[mech])
            r, p = pearsonr(confs, accuracies)
            mechanism_analysis[mech] = {
                'confidence_accuracy_correlation': r,
                'p_value': p,
                'mean_confidence': np.mean(confs),
                'std_confidence': np.std(confs)
            }
        
        return {
            'mechanism_analysis': mechanism_analysis,
            'overall_confidence_accuracy_correlation': pearsonr(overall_confidences, accuracies)[0],
            'mechanisms': mechanisms
        }
    
    def _analyze_metacognition(self, behavioral_results: List[Dict]) -> Dict:
        """Analyze metacognitive accuracy."""
        confidences = np.array([r['confidence'] for r in behavioral_results])
        accuracies = np.array([r['correct'] for r in behavioral_results])
        rts = np.array([r['reaction_time'] for r in behavioral_results])
        
        # Type 2 sensitivity
        meta_results = MetacognitionAnalyzer.calculate_type2_sensitivity(
            confidences, accuracies
        )
        
        # Confidence resolution
        resolution = MetacognitionAnalyzer.analyze_confidence_resolution(
            confidences, accuracies
        )
        
        # Confidence-RT relationship
        r_conf_rt, p_conf_rt = pearsonr(confidences, rts)
        
        return {
            **meta_results,
            'confidence_resolution': resolution,
            'confidence_rt_correlation': r_conf_rt,
            'confidence_rt_p_value': p_conf_rt
        }
    
    def _compare_models(self, behavioral_results: List[Dict], 
                      ddm_results: List[Dict]) -> Dict:
        """Compare spiking network and DDM performance."""
        # Extract data
        snn_choices = np.array([r['choice'] for r in behavioral_results])
        snn_rts = np.array([r['reaction_time'] for r in behavioral_results])
        snn_confidences = np.array([r['confidence'] for r in behavioral_results])
        snn_correct = np.array([r['correct'] for r in behavioral_results])
        
        ddm_choices = np.array([r['choice'] for r in ddm_results])
        ddm_rts = np.array([r['reaction_time'] for r in ddm_results])
        ddm_confidences = np.array([r['confidence'] for r in ddm_results])
        ddm_correct = np.array([r['correct'] for r in ddm_results])
        
        # Calculate performance metrics
        comparison = {
            'accuracy': {
                'snn': np.mean(snn_correct),
                'ddm': np.mean(ddm_correct)
            },
            'mean_rt': {
                'snn': np.mean(snn_rts),
                'ddm': np.mean(ddm_rts)
            },
            'mean_confidence': {
                'snn': np.mean(snn_confidences),
                'ddm': np.mean(ddm_confidences)
            },
            'choice_agreement': np.mean(snn_choices == ddm_choices),
            'rt_correlation': pearsonr(snn_rts, ddm_rts)[0],
            'confidence_correlation': pearsonr(snn_confidences, ddm_confidences)[0]
        }
        
        return comparison
    
    def _save_results(self, results: ExperimentResults):
        """Save experimental results to files."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = self.results_dir / f"{self.config.experiment_name}_{timestamp}"
        experiment_dir.mkdir(exist_ok=True)
        
        # Save behavioral data as CSV
        behavioral_df = pd.DataFrame(results.behavioral_results)
        behavioral_df.to_csv(experiment_dir / "behavioral_data.csv", index=False)
        
        # Save DDM data as CSV
        ddm_df = pd.DataFrame(results.ddm_results)
        ddm_df.to_csv(experiment_dir / "ddm_data.csv", index=False)
        
        # Save analyses as JSON
        analyses = {
            'psychometric': results.psychometric_analysis,
            'confidence': results.confidence_analysis,
            'metacognition': results.metacognition_analysis,
            'model_comparison': results.model_comparison
        }
        
        with open(experiment_dir / "analyses.json", 'w') as f:
            json.dump(analyses, f, indent=2, default=str)
        
        # Save config
        config_dict = asdict(results.config)
        config_dict['network_params'] = asdict(results.config.network_params)
        config_dict['ddm_params'] = asdict(results.config.ddm_params)
        
        with open(experiment_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Generate summary plots
        self._generate_summary_plots(results, experiment_dir)
        
        print(f"Results saved to: {experiment_dir}")
    
    def _generate_summary_plots(self, results: ExperimentResults, save_dir: Path):
        """Generate comprehensive summary plots."""
        # Psychometric and chronometric curves
        fig1 = PsychophysicalAnalysis.plot_psychometric_curve(
            np.array(results.psychometric_analysis['coherences']),
            np.array(results.psychometric_analysis['accuracies']),
            np.array(results.psychometric_analysis['reaction_times'])
        )
        fig1.suptitle(f'{self.config.experiment_name}: Psychophysical Performance')
        fig1.savefig(save_dir / "psychophysical_curves.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Confidence analysis
        confidences = np.array([r['confidence'] for r in results.behavioral_results])
        accuracies = np.array([r['correct'] for r in results.behavioral_results])
        rts = np.array([r['reaction_time'] for r in results.behavioral_results])
        
        fig2 = MetacognitionAnalyzer.plot_confidence_analysis(confidences, accuracies, rts)
        fig2.suptitle(f'{self.config.experiment_name}: Confidence Analysis')
        fig2.savefig(save_dir / "confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # Model comparison
        fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        snn_acc = results.model_comparison['accuracy']['snn']
        ddm_acc = results.model_comparison['accuracy']['ddm']
        
        axes[0, 0].bar(['SNN', 'DDM'], [snn_acc, ddm_acc])
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Model Accuracy Comparison')
        
        snn_rt = results.model_comparison['mean_rt']['snn'] 
        ddm_rt = results.model_comparison['mean_rt']['ddm']
        
        axes[0, 1].bar(['SNN', 'DDM'], [snn_rt, ddm_rt])
        axes[0, 1].set_ylabel('Mean RT (s)')
        axes[0, 1].set_title('Model RT Comparison')
        
        # Confidence mechanisms
        mech_analysis = results.confidence_analysis['mechanism_analysis']
        mechanisms = list(mech_analysis.keys())
        correlations = [mech_analysis[m]['confidence_accuracy_correlation'] for m in mechanisms]
        
        axes[1, 0].barh(range(len(mechanisms)), correlations)
        axes[1, 0].set_yticks(range(len(mechanisms)))
        axes[1, 0].set_yticklabels([m.replace('_', ' ').title() for m in mechanisms])
        axes[1, 0].set_xlabel('Confidence-Accuracy Correlation')
        axes[1, 0].set_title('Confidence Mechanisms')
        
        # Metacognition summary
        meta_sens = results.metacognition_analysis['meta_sensitivity']
        auroc = results.metacognition_analysis['auroc']
        
        axes[1, 1].bar(['Meta-Sensitivity', 'AUROC'], [meta_sens, auroc])
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_title('Metacognitive Performance')
        
        fig3.suptitle(f'{self.config.experiment_name}: Model & Mechanism Summary')
        plt.tight_layout()
        fig3.savefig(save_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)


# Example usage
if __name__ == "__main__":
    # Create experiment configuration
    config = ExperimentConfig(
        experiment_name="neural_confidence_pilot",
        coherence_levels=[0.0, 0.128, 0.256, 0.512],
        n_trials_per_condition=25,  # Reduced for pilot
        network_params=NetworkParams(n_sensory=100, n_decision=60)
    )
    
    # Run experiment
    experiment = NeuralConfidenceExperiment(config)
    results = experiment.run_full_experiment()
    
    # Print summary
    print("\\n=== Experiment Summary ===")
    print(f"Overall accuracy: {results.psychometric_analysis['overall_accuracy']:.3f}")
    print(f"Mean RT: {results.psychometric_analysis['overall_rt']:.3f}s")
    print(f"Metacognitive sensitivity: {results.metacognition_analysis['meta_sensitivity']:.3f}")
    print(f"SNN vs DDM accuracy: {results.model_comparison['accuracy']['snn']:.3f} vs {results.model_comparison['accuracy']['ddm']:.3f}")
    
    print("✓ Experiment framework implementation complete")