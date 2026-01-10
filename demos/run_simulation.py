"""
Brain-Behavior Mapping: Integrated BCI + RL Simulation
======================================================
Person 3 Deliverable: Main integration script

This script integrates:
- Person 1's BCI decoder (motor imagery classification)
- Person 2's RL environment and Actor-Critic agent

The simulation demonstrates a "mind-controlled gambling task" where:
1. EEG signals (real or synthetic) are decoded into intentions (Left/Right)
2. The intention drives actions in a 2-armed bandit environment
3. The agent learns optimal strategies through dopamine-like RPE signals
4. Behavior adapts during reversal learning

Author: Person 3 - Integration, UX & Explainability
Date: January 2026
"""

import sys
import os
import time
import argparse
import pickle
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Tuple, List

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from envs.gambling_task import GamblingTaskEnv
from models.rpe_agent import ActorCriticAgent
from utils.viz import (
    plot_rpe_dynamics,
    plot_reversal_learning,
    plot_reward_history,
    plot_policy_evolution,
    create_dashboard_figure
)

# Optional: BCI decoder (may not be available in all setups)
try:
    from models.bci_decoder import BCIDecoder
    BCI_AVAILABLE = True
except ImportError:
    BCI_AVAILABLE = False
    print("⚠️ BCI decoder not available. Using synthetic/random actions.")

# Optional: XAI utilities with SHAP/MNE/MOABB integration
try:
    from utils.xai import (
        SHAP_AVAILABLE, MNE_AVAILABLE, MOABB_AVAILABLE,
        compute_shap_values, plot_shap_summary,
        plot_mne_topomap, get_dataset_info, load_sample_eeg_data,
        explain_prediction, EEG_CHANNELS
    )
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    SHAP_AVAILABLE = False
    MNE_AVAILABLE = False
    MOABB_AVAILABLE = False


class BrainBehaviorSimulation:
    """
    Main simulation class integrating BCI and RL components.
    
    This class manages the closed-loop simulation where:
    - EEG signals → decoded intentions → actions
    - Actions → rewards → RPE → learning
    
    Parameters
    ----------
    n_trials : int
        Number of trials to run
    reversal_trial : int
        Trial where reward probabilities flip
    p_left : float
        Initial reward probability for left action
    p_right : float
        Initial reward probability for right action
    learning_rate : float
        Agent learning rate
    gamma : float
        Discount factor
    use_bci : bool
        Whether to use BCI decoder for actions (vs random/synthetic)
    bci_confidence_threshold : float
        Minimum BCI confidence to trust the decoded intention
    seed : int
        Random seed for reproducibility
    verbose : bool
        Print progress during simulation
    """
    
    def __init__(
        self,
        n_trials: int = 1000,
        reversal_trial: int = 500,
        p_left: float = 0.4,
        p_right: float = 0.6,
        learning_rate: float = 0.01,
        gamma: float = 0.99,
        temperature: float = 1.0,
        use_bci: bool = False,
        bci_confidence_threshold: float = 0.6,
        seed: int = 42,
        verbose: bool = True
    ):
        self.n_trials = n_trials
        self.reversal_trial = reversal_trial
        self.p_left = p_left
        self.p_right = p_right
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.temperature = temperature
        self.use_bci = use_bci and BCI_AVAILABLE
        self.bci_confidence_threshold = bci_confidence_threshold
        self.seed = seed
        self.verbose = verbose
        
        # Initialize components
        self._init_environment()
        self._init_agent()
        self._init_bci()
        
        # Results storage
        self.results = {
            'rpe_history': [],
            'reward_history': [],
            'action_history': [],
            'policy_history': [],
            'value_history': [],
            'bci_confidence_history': [],
            'decoded_intentions': [],
            'trial_info': []
        }
        
        self.is_running = False
        self.current_trial = 0
        
    def _init_environment(self):
        """Initialize the gambling task environment."""
        self.env = GamblingTaskEnv(
            p_left=self.p_left,
            p_right=self.p_right,
            reversal_trial=self.reversal_trial,
            seed=self.seed
        )
        if self.verbose:
            print(f"✅ Environment initialized: p_left={self.p_left}, p_right={self.p_right}")
            print(f"   Reversal at trial {self.reversal_trial}")
    
    def _init_agent(self):
        """Initialize the Actor-Critic agent."""
        self.agent = ActorCriticAgent(
            state_size=10,  # Match environment state size
            action_size=2,  # Left or Right
            learning_rate=self.learning_rate,
            gamma=self.gamma,
            temperature=self.temperature,
            seed=self.seed
        )
        if self.verbose:
            print(f"✅ Agent initialized: lr={self.learning_rate}, γ={self.gamma}")
    
    def _init_bci(self):
        """Initialize the BCI decoder if available and requested."""
        self.bci_decoder = None
        
        if self.use_bci:
            model_path = PROJECT_ROOT / 'models' / 'shallow_convnet_motor_imagery.keras'
            
            if model_path.exists():
                try:
                    self.bci_decoder = BCIDecoder(str(model_path), verbose=self.verbose)
                    if self.verbose:
                        print(f"✅ BCI decoder loaded from {model_path}")
                except Exception as e:
                    print(f"⚠️ Failed to load BCI decoder: {e}")
                    self.bci_decoder = None
            else:
                print(f"⚠️ BCI model not found at {model_path}")
        
        if self.bci_decoder is None and self.verbose:
            print("ℹ️ Using agent policy for action selection (no BCI)")
        
        # Report XAI library availability
        if self.verbose:
            self._report_xai_status()
    
    def _report_xai_status(self):
        """Report the status of XAI-related libraries."""
        print("\n🔬 XAI/BCI Library Status:")
        
        if XAI_AVAILABLE:
            print(f"   SHAP:  {'✅ Available' if SHAP_AVAILABLE else '❌ Not installed'}")
            print(f"   MNE:   {'✅ Available' if MNE_AVAILABLE else '❌ Not installed'}")
            print(f"   MOABB: {'✅ Available' if MOABB_AVAILABLE else '❌ Not installed'}")
            
            if MOABB_AVAILABLE:
                try:
                    dataset_info = get_dataset_info()
                    print(f"   Dataset: {dataset_info.get('name', 'Unknown')} "
                          f"({dataset_info.get('n_subjects', '?')} subjects, "
                          f"{dataset_info.get('n_channels', '?')} channels)")
                except Exception:
                    pass
        else:
            print("   ❌ XAI utilities not available")
        
        print("")  # Empty line for spacing
    
    def _get_action_from_bci(self, state: np.ndarray) -> Tuple[int, float, str]:
        """
        Get action from BCI decoder using synthetic EEG.
        
        Returns
        -------
        action : int
            Decoded action (0=Left, 1=Right)
        confidence : float
            Decoder confidence
        intention : str
            'Left' or 'Right'
        """
        if self.bci_decoder is None:
            # Fall back to agent policy
            action = self.agent.get_action(state, deterministic=False)
            return action, 1.0, 'Left' if action == 0 else 'Right'
        
        # Generate synthetic EEG based on current policy preference
        # In real BCI, this would be actual EEG data
        policy = self.agent.policy_history[-1] if self.agent.policy_history else [0.5, 0.5]
        preferred_class = 'Left' if policy[0] > policy[1] else 'Right'
        
        synthetic_eeg = self.bci_decoder.generate_synthetic_eeg(
            n_trials=1, 
            class_label=preferred_class
        )
        
        # Get prediction
        result = self.bci_decoder.predict(synthetic_eeg)
        
        action = result['action']
        confidence = result['confidence']
        intention = result['class']
        
        # Apply confidence threshold
        if confidence < self.bci_confidence_threshold:
            # Low confidence: fall back to agent policy
            action = self.agent.get_action(state, deterministic=False)
            intention = 'Left' if action == 0 else 'Right'
        
        return action, confidence, intention
    
    def _get_action_from_agent(self, state: np.ndarray) -> Tuple[int, float, str]:
        """
        Get action directly from agent policy.
        
        Returns
        -------
        action : int
            Selected action
        confidence : float
            Policy probability
        intention : str
            'Left' or 'Right'
        """
        action = self.agent.get_action(state, deterministic=False)
        
        # Get confidence from latest policy
        if self.agent.policy_history:
            confidence = self.agent.policy_history[-1][action]
        else:
            confidence = 0.5
            
        intention = 'Left' if action == 0 else 'Right'
        
        return action, confidence, intention
    
    def step(self, state: np.ndarray) -> Dict:
        """
        Execute a single simulation step.
        
        Parameters
        ----------
        state : np.ndarray
            Current environment state
            
        Returns
        -------
        step_info : dict
            Information about this step
        """
        # Get action (from BCI or agent)
        if self.use_bci and self.bci_decoder is not None:
            action, confidence, intention = self._get_action_from_bci(state)
        else:
            action, confidence, intention = self._get_action_from_agent(state)
        
        # Execute action in environment
        next_state, reward, done, info = self.env.step(action)
        
        # Update agent (learn from experience)
        rpe = self.agent.update(state, action, reward, next_state, done)
        
        # Store results
        step_info = {
            'trial': self.current_trial,
            'action': action,
            'intention': intention,
            'confidence': confidence,
            'reward': reward,
            'rpe': rpe,
            'value': self.agent.value_history[-1] if self.agent.value_history else 0,
            'policy': self.agent.policy_history[-1].copy() if self.agent.policy_history else [0.5, 0.5],
            'reversal_occurred': info.get('reversal_occurred', False),
            'done': done
        }
        
        # Update results
        self.results['rpe_history'].append(rpe)
        self.results['reward_history'].append(reward)
        self.results['action_history'].append(action)
        self.results['policy_history'].append(step_info['policy'])
        self.results['value_history'].append(step_info['value'])
        self.results['bci_confidence_history'].append(confidence)
        self.results['decoded_intentions'].append(intention)
        self.results['trial_info'].append(info)
        
        self.current_trial += 1
        
        return step_info, next_state, done
    
    def run(self, callback=None) -> Dict:
        """
        Run the full simulation.
        
        Parameters
        ----------
        callback : callable, optional
            Function called after each trial: callback(step_info, trial_num)
            
        Returns
        -------
        results : dict
            Complete simulation results
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print("🧠 STARTING BRAIN-BEHAVIOR SIMULATION")
            print(f"{'='*60}")
            print(f"Trials: {self.n_trials} | Reversal: {self.reversal_trial}")
            print(f"BCI mode: {'ON' if self.use_bci else 'OFF'}")
            print(f"{'='*60}\n")
        
        self.is_running = True
        state = self.env.reset()
        
        start_time = time.time()
        
        for trial in range(self.n_trials):
            step_info, state, done = self.step(state)
            
            # Progress callback
            if callback is not None:
                callback(step_info, trial)
            
            # Print progress
            if self.verbose and (trial + 1) % 100 == 0:
                avg_reward = np.mean(self.results['reward_history'][-100:])
                avg_rpe = np.mean(self.results['rpe_history'][-100:])
                print(f"Trial {trial+1:4d} | Reward rate: {avg_reward:.1%} | "
                      f"Avg RPE: {avg_rpe:+.4f}")
            
            # Check for reversal
            if trial == self.reversal_trial and self.verbose:
                print(f"\n🔄 REVERSAL at trial {trial}! Probabilities flipped.\n")
            
            if done:
                state = self.env.reset()
        
        self.is_running = False
        elapsed = time.time() - start_time
        
        # Convert lists to arrays
        for key in ['rpe_history', 'reward_history', 'action_history', 
                    'policy_history', 'value_history', 'bci_confidence_history']:
            self.results[key] = np.array(self.results[key])
        
        # Add summary statistics
        self.results['summary'] = self._compute_summary()
        self.results['config'] = {
            'n_trials': self.n_trials,
            'reversal_trial': self.reversal_trial,
            'p_left': self.p_left,
            'p_right': self.p_right,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'seed': self.seed,
            'use_bci': self.use_bci,
            'elapsed_time': elapsed
        }
        
        if self.verbose:
            self._print_summary()
        
        return self.results
    
    def _compute_summary(self) -> Dict:
        """Compute summary statistics."""
        rpe = self.results['rpe_history']
        rewards = self.results['reward_history']
        actions = self.results['action_history']
        
        # Pre-reversal stats
        pre_rpe = rpe[:self.reversal_trial]
        pre_rewards = rewards[:self.reversal_trial]
        pre_actions = actions[:self.reversal_trial]
        
        # Post-reversal stats
        post_rpe = rpe[self.reversal_trial:]
        post_rewards = rewards[self.reversal_trial:]
        post_actions = actions[self.reversal_trial:]
        
        return {
            'total_reward': np.sum(rewards),
            'overall_reward_rate': np.mean(rewards),
            'pre_reversal_reward_rate': np.mean(pre_rewards),
            'post_reversal_reward_rate': np.mean(post_rewards),
            'mean_rpe': np.mean(rpe),
            'pre_reversal_mean_rpe': np.mean(pre_rpe),
            'post_reversal_mean_rpe': np.mean(post_rpe),
            'pre_reversal_right_rate': np.mean(pre_actions),
            'post_reversal_right_rate': np.mean(post_actions),
            'final_policy': self.results['policy_history'][-1].tolist()
        }
    
    def _print_summary(self):
        """Print simulation summary."""
        s = self.results['summary']
        c = self.results['config']
        
        print(f"\n{'='*60}")
        print("📊 SIMULATION COMPLETE - SUMMARY")
        print(f"{'='*60}")
        print(f"Total trials: {c['n_trials']} | Time: {c['elapsed_time']:.2f}s")
        print(f"\n💰 REWARDS:")
        print(f"  Total: {s['total_reward']}/{c['n_trials']}")
        print(f"  Overall rate: {s['overall_reward_rate']:.1%}")
        print(f"  Pre-reversal: {s['pre_reversal_reward_rate']:.1%}")
        print(f"  Post-reversal: {s['post_reversal_reward_rate']:.1%}")
        print(f"\n⚡ RPE (Dopamine Signal):")
        print(f"  Mean: {s['mean_rpe']:+.4f}")
        print(f"  Pre-reversal: {s['pre_reversal_mean_rpe']:+.4f}")
        print(f"  Post-reversal: {s['post_reversal_mean_rpe']:+.4f}")
        print(f"\n🎯 FINAL POLICY:")
        print(f"  P(Left): {s['final_policy'][0]:.1%}")
        print(f"  P(Right): {s['final_policy'][1]:.1%}")
        print(f"{'='*60}\n")
    
    def save_results(self, path: Optional[str] = None) -> str:
        """
        Save simulation results to a pickle file.
        
        Parameters
        ----------
        path : str, optional
            Save path. If None, auto-generates based on timestamp.
            
        Returns
        -------
        path : str
            Path where results were saved
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = PROJECT_ROOT / 'results' / f'simulation_results_{timestamp}.pkl'
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"✅ Results saved to {path}")
        return str(path)
    
    def plot_results(self, save_dir: Optional[str] = None, show: bool = True):
        """
        Generate all visualization plots.
        
        Parameters
        ----------
        save_dir : str, optional
            Directory to save plots
        show : bool
            Whether to display plots
        """
        import matplotlib.pyplot as plt
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. RPE Dynamics
        fig1 = plot_rpe_dynamics(
            self.results['rpe_history'],
            reversal_trial=self.reversal_trial,
            save_path=str(save_dir / 'rpe_dynamics.png') if save_dir else None
        )
        
        # 2. Reversal Learning Analysis
        fig2 = plot_reversal_learning(
            self.results['rpe_history'],
            self.results['reward_history'],
            self.results['action_history'],
            reversal_trial=self.reversal_trial,
            save_path=str(save_dir / 'reversal_learning.png') if save_dir else None
        )
        
        # 3. Reward History
        fig3 = plot_reward_history(
            self.results['reward_history'],
            self.results['action_history'],
            reversal_trial=self.reversal_trial,
            save_path=str(save_dir / 'reward_history.png') if save_dir else None
        )
        
        # 4. Policy Evolution
        fig4 = plot_policy_evolution(
            self.results['policy_history'],
            reversal_trial=self.reversal_trial,
            save_path=str(save_dir / 'policy_evolution.png') if save_dir else None
        )
        
        if show:
            plt.show()
        
        return [fig1, fig2, fig3, fig4]


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description='Brain-Behavior Mapping Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_simulation.py --trials 1000 --reversal 500
  python run_simulation.py --use-bci --verbose
  python run_simulation.py --save-results --plot
        """
    )
    
    parser.add_argument('--trials', type=int, default=1000,
                       help='Number of trials (default: 1000)')
    parser.add_argument('--reversal', type=int, default=500,
                       help='Reversal trial (default: 500)')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--use-bci', action='store_true',
                       help='Use BCI decoder for actions')
    parser.add_argument('--save-results', action='store_true',
                       help='Save results to pickle file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate and show plots')
    parser.add_argument('--save-plots', type=str, default=None,
                       help='Directory to save plots')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Create and run simulation
    sim = BrainBehaviorSimulation(
        n_trials=args.trials,
        reversal_trial=args.reversal,
        learning_rate=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        use_bci=args.use_bci,
        verbose=not args.quiet
    )
    
    results = sim.run()
    
    # Save results
    if args.save_results:
        sim.save_results()
    
    # Generate plots
    if args.plot or args.save_plots:
        sim.plot_results(save_dir=args.save_plots, show=args.plot)
    
    return results


if __name__ == "__main__":
    # If run directly, execute simulation
    results = main()
