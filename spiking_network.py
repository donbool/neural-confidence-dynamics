# spiking_network.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


@dataclass
class NetworkParams:
    """Parameters for spiking neural network."""
    # Network size
    n_sensory: int = 200      # MT/MST neurons
    n_decision: int = 100     # LIP neurons (50 left, 50 right)
    n_confidence: int = 50    # Confidence area
    n_inhibitory: int = 50    # Interneurons
    
    # Neuron parameters
    tau_m: float = 0.020      # Membrane time constant (20ms)
    tau_syn: float = 0.005    # Synaptic time constant (5ms)  
    v_thresh: float = -50.0   # Spike threshold (mV)
    v_reset: float = -70.0    # Reset potential (mV)
    v_rest: float = -70.0     # Resting potential (mV)
    
    # Synaptic weights (mV)
    w_sensory_decision: float = 0.8
    w_decision_decision: float = 1.2  # Recurrent excitation
    w_decision_confidence: float = 0.6
    w_inhibitory: float = -2.0
    
    # Connection probabilities
    p_sensory_decision: float = 0.3
    p_decision_recurrent: float = 0.2
    p_decision_confidence: float = 0.4
    p_inhibitory: float = 0.3
    
    # Simulation
    dt: float = 0.0001        # 0.1ms time steps
    noise_std: float = 0.5    # Background noise


class LeakyIntegrateFireNetwork:
    """
    Biologically plausible spiking neural network for perceptual decision-making.
    
    Architecture:
    - Sensory layer (MT/MST): Processes motion evidence
    - Decision layer (LIP): Two competing populations (left/right)  
    - Confidence layer: Monitors decision dynamics
    - Inhibitory interneurons: Provide competition and gain control
    """
    
    def __init__(self, params: NetworkParams = None):
        self.params = params or NetworkParams()
        self.setup_network()
        self.reset_state()
    
    def setup_network(self):
        """Initialize network architecture and connectivity."""
        p = self.params
        
        # Total neurons
        self.n_total = (p.n_sensory + p.n_decision + 
                       p.n_confidence + p.n_inhibitory)
        
        # Neuron indices
        self.sensory_idx = slice(0, p.n_sensory)
        self.decision_idx = slice(p.n_sensory, p.n_sensory + p.n_decision)
        self.confidence_idx = slice(p.n_sensory + p.n_decision, 
                                   p.n_sensory + p.n_decision + p.n_confidence)
        self.inhibitory_idx = slice(p.n_sensory + p.n_decision + p.n_confidence, 
                                   self.n_total)
        
        # Decision populations
        self.left_pop = slice(p.n_sensory, p.n_sensory + p.n_decision // 2)
        self.right_pop = slice(p.n_sensory + p.n_decision // 2, 
                              p.n_sensory + p.n_decision)
        
        # Build connectivity matrix
        self.build_connectivity()
        
        print(f"Network initialized:")
        print(f"  Total neurons: {self.n_total}")
        print(f"  Sensory: {p.n_sensory}, Decision: {p.n_decision}")
        print(f"  Confidence: {p.n_confidence}, Inhibitory: {p.n_inhibitory}")
    
    def build_connectivity(self):
        """Build sparse connectivity matrix following Dale's principle."""
        p = self.params
        n = self.n_total
        
        # Initialize connectivity matrix
        W = np.zeros((n, n))
        
        # Sensory → Decision connections
        sensory_neurons = np.arange(p.n_sensory)
        
        # Left-preferring sensory neurons (first half) → Left decision
        left_sensory = sensory_neurons[:p.n_sensory // 2]
        left_decision = np.arange(p.n_sensory, p.n_sensory + p.n_decision // 2)
        
        for i in left_sensory:
            targets = np.random.choice(left_decision, 
                                     size=int(len(left_decision) * p.p_sensory_decision),
                                     replace=False)
            W[targets, i] = p.w_sensory_decision
        
        # Right-preferring sensory neurons → Right decision
        right_sensory = sensory_neurons[p.n_sensory // 2:]
        right_decision = np.arange(p.n_sensory + p.n_decision // 2, 
                                  p.n_sensory + p.n_decision)
        
        for i in right_sensory:
            targets = np.random.choice(right_decision,
                                     size=int(len(right_decision) * p.p_sensory_decision),
                                     replace=False)
            W[targets, i] = p.w_sensory_decision
        
        # Decision → Decision (competitive dynamics)
        decision_neurons = np.arange(p.n_sensory, p.n_sensory + p.n_decision)
        
        for i in decision_neurons:
            # Self-excitation within population
            if i in left_decision:
                same_pop = left_decision
            else:
                same_pop = right_decision
            
            # Connect to same population (excitatory)
            targets = np.random.choice(same_pop,
                                     size=int(len(same_pop) * p.p_decision_recurrent),
                                     replace=False)
            targets = targets[targets != i]  # No self-connections
            W[targets, i] = p.w_decision_decision
        
        # Decision → Confidence
        confidence_neurons = np.arange(p.n_sensory + p.n_decision,
                                     p.n_sensory + p.n_decision + p.n_confidence)
        
        for i in decision_neurons:
            targets = np.random.choice(confidence_neurons,
                                     size=int(len(confidence_neurons) * p.p_decision_confidence),
                                     replace=False)
            W[targets, i] = p.w_decision_confidence
        
        # Inhibitory connections (interneurons)
        inhibitory_neurons = np.arange(p.n_sensory + p.n_decision + p.n_confidence,
                                     self.n_total)
        
        # Decision neurons drive inhibitory neurons
        for i in decision_neurons:
            targets = np.random.choice(inhibitory_neurons,
                                     size=int(len(inhibitory_neurons) * p.p_inhibitory),
                                     replace=False)
            W[targets, i] = p.w_decision_confidence  # Excite inhibitory neurons
        
        # Inhibitory neurons inhibit decision neurons (cross-inhibition)
        for i in inhibitory_neurons:
            targets = np.random.choice(decision_neurons,
                                     size=int(len(decision_neurons) * p.p_inhibitory),
                                     replace=False)
            W[targets, i] = p.w_inhibitory  # Negative weights
        
        # Convert to sparse matrix for efficiency
        self.W = csr_matrix(W)
        
        print(f"Connectivity built: {np.count_nonzero(W)} connections")
    
    def reset_state(self):
        """Reset network state."""
        p = self.params
        
        # Membrane potentials
        self.V = np.full(self.n_total, p.v_rest)
        
        # Synaptic currents
        self.I_syn = np.zeros(self.n_total)
        
        # Spike history
        self.spikes = []
        self.spike_times = []
        
        # External currents
        self.I_ext = np.zeros(self.n_total)
    
    def set_external_input(self, left_evidence: float, right_evidence: float):
        """
        Set external input currents based on sensory evidence.
        
        Args:
            left_evidence: Evidence for leftward motion
            right_evidence: Evidence for rightward motion
        """
        p = self.params
        
        # Reset external input
        self.I_ext.fill(0)
        
        # Drive left-preferring sensory neurons
        left_sensory_end = p.n_sensory // 2
        self.I_ext[:left_sensory_end] = left_evidence
        
        # Drive right-preferring sensory neurons  
        self.I_ext[left_sensory_end:p.n_sensory] = right_evidence
        
        # Add background noise to all neurons
        noise = np.random.normal(0, p.noise_std, self.n_total)
        self.I_ext += noise
    
    def update_neurons(self, dt: float):
        """
        Update neuron states using Euler integration.
        
        Args:
            dt: Time step (seconds)
        """
        p = self.params
        
        # Update synaptic currents (exponential decay)
        self.I_syn *= np.exp(-dt / p.tau_syn)
        
        # Total input current
        I_total = self.I_ext + self.I_syn
        
        # Update membrane potentials
        dV = (-(self.V - p.v_rest) + I_total) / p.tau_m
        self.V += dV * dt
        
        # Find neurons that spiked
        spiked = self.V >= p.v_thresh
        spike_indices = np.where(spiked)[0]
        
        if len(spike_indices) > 0:
            # Reset spiked neurons
            self.V[spiked] = p.v_reset
            
            # Add synaptic input from spikes
            # This is where spikes propagate through network
            for idx in spike_indices:
                # Get postsynaptic targets
                targets = self.W.getcol(idx).nonzero()[0]
                weights = self.W[targets, idx].A1  # Convert to 1D array
                
                # Add synaptic current
                self.I_syn[targets] += weights
            
            return spike_indices
        
        return np.array([])
    
    def simulate_trial(self, stimulus_sequence: np.ndarray, 
                      dt: Optional[float] = None) -> Dict:
        """
        Simulate network response to stimulus sequence.
        
        Args:
            stimulus_sequence: (n_steps, 2) array of [left_evidence, right_evidence]
            dt: Time step (uses network default if None)
            
        Returns:
            results: Dictionary with spike trains and population activities
        """
        if dt is None:
            dt = self.params.dt
        
        n_steps = len(stimulus_sequence)
        
        # Storage for results
        spike_trains = []
        population_rates = {
            'sensory': np.zeros(n_steps),
            'left_decision': np.zeros(n_steps),
            'right_decision': np.zeros(n_steps), 
            'confidence': np.zeros(n_steps)
        }
        
        # Reset network
        self.reset_state()
        
        # Simulation loop
        for t in range(n_steps):
            # Set external input for this time step
            left_ev, right_ev = stimulus_sequence[t]
            self.set_external_input(left_ev, right_ev)
            
            # Update network
            spikes = self.update_neurons(dt)
            spike_trains.append(spikes)
            
            # Calculate population firing rates (sliding window)
            window_size = int(0.050 / dt)  # 50ms window
            start_idx = max(0, t - window_size)
            
            # Count spikes in recent window
            recent_spikes = []
            for s_idx in range(start_idx, t + 1):
                if s_idx < len(spike_trains):
                    recent_spikes.extend(spike_trains[s_idx])
            
            if recent_spikes:
                recent_spikes = np.array(recent_spikes)
                
                # Sensory population rate
                sensory_spikes = recent_spikes[recent_spikes < self.params.n_sensory]
                population_rates['sensory'][t] = len(sensory_spikes) / (window_size * dt * self.params.n_sensory)
                
                # Left decision population
                left_spikes = recent_spikes[(recent_spikes >= self.left_pop.start) & 
                                          (recent_spikes < self.left_pop.stop)]
                population_rates['left_decision'][t] = len(left_spikes) / (window_size * dt * (self.params.n_decision // 2))
                
                # Right decision population  
                right_spikes = recent_spikes[(recent_spikes >= self.right_pop.start) & 
                                           (recent_spikes < self.right_pop.stop)]
                population_rates['right_decision'][t] = len(right_spikes) / (window_size * dt * (self.params.n_decision // 2))
                
                # Confidence population
                conf_start = self.confidence_idx.start
                conf_stop = self.confidence_idx.stop
                conf_spikes = recent_spikes[(recent_spikes >= conf_start) & 
                                          (recent_spikes < conf_stop)]
                population_rates['confidence'][t] = len(conf_spikes) / (window_size * dt * self.params.n_confidence)
        
        return {
            'spike_trains': spike_trains,
            'population_rates': population_rates,
            'final_voltages': self.V.copy(),
            'stimulus': stimulus_sequence
        }
    
    def extract_decision_variables(self, results: Dict) -> Dict:
        """
        Extract decision variables from network activity.
        
        Args:
            results: Results from simulate_trial()
            
        Returns:
            decision_vars: Decision-related variables
        """
        rates = results['population_rates']
        
        # Decision variable (difference between populations)
        decision_variable = rates['left_decision'] - rates['right_decision']
        
        # Choice (based on final decision variable)
        final_dv = decision_variable[-100:].mean()  # Average last 100ms
        choice = 0 if final_dv < 0 else 1  # 0=left, 1=right
        
        # Reaction time (first threshold crossing)
        threshold = 5.0  # spikes/s threshold
        rt_idx = None
        
        for t, dv in enumerate(np.abs(decision_variable)):
            if dv > threshold:
                rt_idx = t
                break
        
        reaction_time = (rt_idx * self.params.dt) if rt_idx else len(decision_variable) * self.params.dt
        
        # Confidence metrics
        confidence_rate = rates['confidence'].mean()
        
        # Decision variable variance (uncertainty measure)
        dv_variance = np.var(decision_variable)
        
        # Time to decision stability
        decision_stability = self._calculate_stability_time(decision_variable)
        
        return {
            'decision_variable': decision_variable,
            'choice': choice,
            'reaction_time': reaction_time,
            'confidence_rate': confidence_rate,
            'dv_variance': dv_variance,
            'decision_stability': decision_stability,
            'choice_certainty': np.abs(final_dv)
        }
    
    def _calculate_stability_time(self, decision_variable: np.ndarray, 
                                window_ms: float = 100) -> float:
        """
        Calculate time when decision becomes stable.
        
        Args:
            decision_variable: Time series of decision variable
            window_ms: Window size for stability criterion (ms)
            
        Returns:
            stability_time: Time to stability (seconds)
        """
        window_size = int(window_ms / (self.params.dt * 1000))
        
        for t in range(window_size, len(decision_variable)):
            window = decision_variable[t-window_size:t]
            # Check if all values in window have same sign
            if np.all(window >= 0) or np.all(window <= 0):
                return t * self.params.dt
        
        return len(decision_variable) * self.params.dt


# Example usage and testing
if __name__ == "__main__":
    # Create network
    print("Initializing spiking neural network...")
    params = NetworkParams(n_sensory=100, n_decision=60, n_confidence=30)
    network = LeakyIntegrateFireNetwork(params)
    
    # Create sample stimulus
    duration = 1.0  # 1 second
    dt = 0.001  # 1ms
    n_steps = int(duration / dt)
    
    # High coherence rightward motion
    coherence = 0.3
    left_evidence = 2.0 - coherence * 2.0
    right_evidence = 2.0 + coherence * 2.0
    
    stimulus = np.zeros((n_steps, 2))
    stimulus[:, 0] = left_evidence + np.random.normal(0, 0.5, n_steps)
    stimulus[:, 1] = right_evidence + np.random.normal(0, 0.5, n_steps)
    stimulus = np.maximum(stimulus, 0.1)  # Non-negative
    
    print(f"Simulating trial (coherence={coherence:.2f})...")
    results = network.simulate_trial(stimulus, dt)
    
    # Extract decision variables
    decision_vars = network.extract_decision_variables(results)
    
    print(f"Choice: {'Right' if decision_vars['choice'] == 1 else 'Left'}")
    print(f"Reaction time: {decision_vars['reaction_time']:.3f}s")
    print(f"Confidence rate: {decision_vars['confidence_rate']:.1f} Hz")
    print(f"Choice certainty: {decision_vars['choice_certainty']:.2f}")
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    time = np.arange(n_steps) * dt
    
    # Population firing rates
    axes[0].plot(time, results['population_rates']['left_decision'], 
                label='Left Decision', linewidth=2)
    axes[0].plot(time, results['population_rates']['right_decision'], 
                label='Right Decision', linewidth=2)
    axes[0].plot(time, results['population_rates']['confidence'], 
                label='Confidence', linewidth=2)
    axes[0].set_ylabel('Firing Rate (Hz)')
    axes[0].set_title('Population Activities')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Decision variable
    axes[1].plot(time, decision_vars['decision_variable'], 'k-', linewidth=2)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(decision_vars['reaction_time'], color='red', linestyle='--', 
                   label=f'RT = {decision_vars["reaction_time"]:.3f}s')
    axes[1].set_ylabel('Decision Variable')
    axes[1].set_title('Decision Variable (Left - Right)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Stimulus
    axes[2].plot(time[:500], stimulus[:500, 0], label='Left Evidence', alpha=0.7)
    axes[2].plot(time[:500], stimulus[:500, 1], label='Right Evidence', alpha=0.7)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Evidence')
    axes[2].set_title('Stimulus (first 500ms)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("✓ Spiking neural network implementation complete")