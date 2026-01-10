"""
Visualization Utilities for Brain-Behavior Mapping Project
===========================================================
Person 3 Deliverable: RPE dynamics, reversal learning, reward history plots

Author: Person 3 - Integration, UX & Explainability
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import List, Optional, Dict, Tuple, Union
import warnings

# Configure matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_rpe_dynamics(
    rpe_history: np.ndarray,
    reversal_trial: Optional[int] = None,
    window_size: int = 20,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Reward Prediction Error (RPE) Dynamics",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot RPE dynamics over trials, showing phasic dopamine response patterns.
    
    Parameters
    ----------
    rpe_history : np.ndarray
        Array of RPE values (δ) for each trial
    reversal_trial : int, optional
        Trial number where reward probabilities reversed
    window_size : int, default=20
        Window size for rolling average smoothing
    figsize : tuple, default=(12, 6)
        Figure size
    title : str
        Plot title
    save_path : str, optional
        Path to save the figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    trials = np.arange(len(rpe_history))
    
    # Top: Raw RPE with color coding
    ax1 = axes[0]
    colors = ['#e74c3c' if r < 0 else '#27ae60' for r in rpe_history]
    ax1.bar(trials, rpe_history, color=colors, alpha=0.6, width=1.0)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    if reversal_trial is not None:
        ax1.axvline(x=reversal_trial, color='#9b59b6', linestyle='--', 
                   linewidth=2, label=f'Reversal (trial {reversal_trial})')
    
    ax1.set_ylabel('RPE (δ)', fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Add interpretation annotations
    ax1.annotate('Positive RPE\n(Better than expected)', 
                xy=(0.02, 0.95), xycoords='axes fraction',
                fontsize=9, color='#27ae60', va='top')
    ax1.annotate('Negative RPE\n(Worse than expected)', 
                xy=(0.02, 0.05), xycoords='axes fraction',
                fontsize=9, color='#e74c3c', va='bottom')
    
    # Bottom: Smoothed RPE trend
    ax2 = axes[1]
    if len(rpe_history) >= window_size:
        rolling_mean = np.convolve(rpe_history, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        rolling_std = np.array([np.std(rpe_history[max(0, i-window_size):i+1]) 
                               for i in range(len(rpe_history))])
        x_smooth = np.arange(window_size-1, len(rpe_history))
        
        ax2.fill_between(x_smooth, 
                        rolling_mean - rolling_std[window_size-1:],
                        rolling_mean + rolling_std[window_size-1:],
                        alpha=0.3, color='#3498db')
        ax2.plot(x_smooth, rolling_mean, color='#3498db', linewidth=2,
                label=f'Rolling mean (window={window_size})')
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    if reversal_trial is not None:
        ax2.axvline(x=reversal_trial, color='#9b59b6', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Trial', fontsize=11)
    ax2.set_ylabel('Smoothed RPE', fontsize=11)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    return fig


def plot_reversal_learning(
    rpe_history: np.ndarray,
    reward_history: np.ndarray,
    action_history: np.ndarray,
    reversal_trial: int,
    window_size: int = 50,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize behavioral adaptation during reversal learning.
    
    Shows how the agent adapts when reward probabilities flip,
    demonstrating dopamine-driven learning and extinction.
    
    Parameters
    ----------
    rpe_history : np.ndarray
        RPE values for each trial
    reward_history : np.ndarray
        Rewards received (0 or 1)
    action_history : np.ndarray
        Actions taken (0=Left, 1=Right)
    reversal_trial : int
        Trial where probabilities reversed
    window_size : int, default=50
        Window for computing rolling statistics
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1])
    
    trials = np.arange(len(rpe_history))
    
    # 1. RPE before and after reversal
    ax1 = fig.add_subplot(gs[0, :])
    
    # Pre-reversal
    pre_rpe = rpe_history[:reversal_trial]
    post_rpe = rpe_history[reversal_trial:]
    
    ax1.fill_between(range(len(pre_rpe)), pre_rpe, alpha=0.5, color='#3498db', label='Pre-reversal')
    ax1.fill_between(range(reversal_trial, len(rpe_history)), post_rpe, alpha=0.5, color='#e74c3c', label='Post-reversal')
    ax1.axvline(x=reversal_trial, color='black', linestyle='--', linewidth=2, label='Reversal point')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('RPE (δ)', fontsize=11)
    ax1.set_title('🧠 Reward Prediction Error: Pre vs Post Reversal', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # 2. Action probability over time (choice behavior)
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Compute rolling action probability (probability of choosing Right)
    rolling_right = np.convolve(action_history, np.ones(window_size)/window_size, mode='valid')
    x_roll = np.arange(window_size-1, len(action_history))
    
    ax2.plot(x_roll, rolling_right, color='#9b59b6', linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    ax2.axvline(x=reversal_trial, color='black', linestyle='--', linewidth=2)
    ax2.fill_between(x_roll, 0.5, rolling_right, where=(rolling_right > 0.5), 
                    alpha=0.3, color='#27ae60', label='Prefer Right')
    ax2.fill_between(x_roll, rolling_right, 0.5, where=(rolling_right < 0.5), 
                    alpha=0.3, color='#e74c3c', label='Prefer Left')
    ax2.set_xlabel('Trial', fontsize=10)
    ax2.set_ylabel('P(Right)', fontsize=10)
    ax2.set_title('Choice Behavior Over Time', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='best', fontsize=8)
    
    # 3. Reward rate over time
    ax3 = fig.add_subplot(gs[1, 1])
    
    rolling_reward = np.convolve(reward_history, np.ones(window_size)/window_size, mode='valid')
    ax3.plot(x_roll, rolling_reward, color='#f39c12', linewidth=2)
    ax3.axvline(x=reversal_trial, color='black', linestyle='--', linewidth=2)
    ax3.axhline(y=0.6, color='#27ae60', linestyle=':', alpha=0.7, label='Optimal rate (60%)')
    ax3.set_xlabel('Trial', fontsize=10)
    ax3.set_ylabel('Reward Rate', fontsize=10)
    ax3.set_title('Reward Rate Over Time', fontsize=11)
    ax3.set_ylim(0, 1)
    ax3.legend(loc='best', fontsize=8)
    
    # 4. Box plots comparing pre vs post reversal
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Adaptation analysis: Compare early post-reversal vs late post-reversal
    early_post = rpe_history[reversal_trial:reversal_trial+100] if len(rpe_history) > reversal_trial+100 else rpe_history[reversal_trial:]
    late_post = rpe_history[-100:] if len(rpe_history) > 100 else rpe_history[reversal_trial:]
    
    data_box = [pre_rpe, early_post, late_post]
    labels_box = ['Pre-reversal', 'Early post\n(adaptation)', 'Late post\n(re-learned)']
    colors_box = ['#3498db', '#e74c3c', '#27ae60']
    
    bp = ax4.boxplot(data_box, labels=labels_box, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax4.set_ylabel('RPE Distribution', fontsize=10)
    ax4.set_title('RPE Distribution Across Phases', fontsize=11)
    
    # 5. Learning metrics summary
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')
    
    # Calculate metrics
    pre_reward_rate = np.mean(reward_history[:reversal_trial])
    post_reward_rate = np.mean(reward_history[reversal_trial:])
    pre_rpe_mean = np.mean(np.abs(rpe_history[:reversal_trial]))
    post_rpe_mean = np.mean(np.abs(rpe_history[reversal_trial:]))
    
    # Compute adaptation speed (trials to reach 55% accuracy after reversal)
    post_rolling = np.convolve(reward_history[reversal_trial:], 
                               np.ones(20)/20, mode='valid')
    adaptation_trials = np.argmax(post_rolling > 0.55) if np.any(post_rolling > 0.55) else len(post_rolling)
    
    metrics_text = f"""
    📊 LEARNING METRICS SUMMARY
    {'='*40}
    
    PRE-REVERSAL (trials 0-{reversal_trial}):
      • Reward Rate: {pre_reward_rate:.1%}
      • Mean |RPE|: {pre_rpe_mean:.3f}
    
    POST-REVERSAL (trials {reversal_trial}+):
      • Reward Rate: {post_reward_rate:.1%}
      • Mean |RPE|: {post_rpe_mean:.3f}
    
    ADAPTATION:
      • Trials to adapt: ~{adaptation_trials}
      • RPE spike at reversal: {'Yes ✓' if np.abs(rpe_history[reversal_trial]) > pre_rpe_mean * 1.5 else 'No'}
    
    {'='*40}
    """
    
    ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    return fig


def plot_reward_history(
    reward_history: np.ndarray,
    action_history: Optional[np.ndarray] = None,
    reversal_trial: Optional[int] = None,
    window_size: int = 20,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cumulative reward and rolling reward rate.
    
    Parameters
    ----------
    reward_history : np.ndarray
        Rewards received each trial (0 or 1)
    action_history : np.ndarray, optional
        Actions taken each trial
    reversal_trial : int, optional
        Trial where reversal occurred
    window_size : int
        Window for rolling average
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    trials = np.arange(len(reward_history))
    
    # Left: Cumulative reward
    ax1 = axes[0]
    cumulative = np.cumsum(reward_history)
    ax1.plot(trials, cumulative, color='#27ae60', linewidth=2)
    ax1.fill_between(trials, 0, cumulative, alpha=0.3, color='#27ae60')
    
    # Add optimal line (60% reward rate)
    optimal = trials * 0.6
    ax1.plot(trials, optimal, color='gray', linestyle='--', alpha=0.7, label='Optimal (60%)')
    
    if reversal_trial is not None:
        ax1.axvline(x=reversal_trial, color='#9b59b6', linestyle='--', linewidth=2)
    
    ax1.set_xlabel('Trial', fontsize=11)
    ax1.set_ylabel('Cumulative Reward', fontsize=11)
    ax1.set_title('💰 Cumulative Reward Over Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    
    # Right: Rolling reward rate
    ax2 = axes[1]
    if len(reward_history) >= window_size:
        rolling_rate = np.convolve(reward_history, 
                                   np.ones(window_size)/window_size, 
                                   mode='valid')
        x_roll = np.arange(window_size-1, len(reward_history))
        
        ax2.plot(x_roll, rolling_rate, color='#3498db', linewidth=2)
        ax2.fill_between(x_roll, 0.5, rolling_rate, 
                        where=(rolling_rate > 0.5), alpha=0.3, color='#27ae60')
        ax2.fill_between(x_roll, rolling_rate, 0.5, 
                        where=(rolling_rate < 0.5), alpha=0.3, color='#e74c3c')
    
    ax2.axhline(y=0.6, color='#27ae60', linestyle='--', alpha=0.7, label='Optimal (60%)')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Chance (50%)')
    
    if reversal_trial is not None:
        ax2.axvline(x=reversal_trial, color='#9b59b6', linestyle='--', linewidth=2,
                   label='Reversal')
    
    ax2.set_xlabel('Trial', fontsize=11)
    ax2.set_ylabel('Reward Rate', fontsize=11)
    ax2.set_title(f'📈 Rolling Reward Rate (window={window_size})', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    return fig


def plot_policy_evolution(
    policy_history: np.ndarray,
    reversal_trial: Optional[int] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize how the agent's policy evolves over training.
    
    Parameters
    ----------
    policy_history : np.ndarray
        Array of shape (n_trials, 2) with P(Left) and P(Right) per trial
    reversal_trial : int, optional
        Trial where reversal occurred
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    policy_history = np.array(policy_history)
    trials = np.arange(len(policy_history))
    
    # Plot probabilities
    ax.plot(trials, policy_history[:, 0], color='#e74c3c', linewidth=2, 
           label='P(Left)', alpha=0.8)
    ax.plot(trials, policy_history[:, 1], color='#3498db', linewidth=2, 
           label='P(Right)', alpha=0.8)
    
    # Fill between to show preference
    ax.fill_between(trials, policy_history[:, 0], policy_history[:, 1],
                   where=(policy_history[:, 1] > policy_history[:, 0]),
                   alpha=0.2, color='#3498db', label='Prefer Right')
    ax.fill_between(trials, policy_history[:, 0], policy_history[:, 1],
                   where=(policy_history[:, 0] > policy_history[:, 1]),
                   alpha=0.2, color='#e74c3c', label='Prefer Left')
    
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    if reversal_trial is not None:
        ax.axvline(x=reversal_trial, color='#9b59b6', linestyle='--', 
                  linewidth=2, label='Reversal')
        
        # Add annotations
        ax.annotate('Pre-reversal:\nRight is better (60%)', 
                   xy=(reversal_trial/4, 0.85), fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='#3498db', alpha=0.3))
        ax.annotate('Post-reversal:\nLeft is better (60%)', 
                   xy=(reversal_trial + (len(trials)-reversal_trial)/2, 0.85), 
                   fontsize=9, ha='center',
                   bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.3))
    
    ax.set_xlabel('Trial', fontsize=11)
    ax.set_ylabel('Action Probability', fontsize=11)
    ax.set_title('🎯 Policy Evolution: P(Left) vs P(Right)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(loc='center right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    return fig


def plot_action_distribution(
    action_history: np.ndarray,
    reward_history: np.ndarray,
    reversal_trial: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot action distribution and reward rates per action.
    
    Parameters
    ----------
    action_history : np.ndarray
        Actions taken (0=Left, 1=Right)
    reward_history : np.ndarray
        Rewards received
    reversal_trial : int, optional
        Trial where reversal occurred
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    action_history = np.array(action_history)
    reward_history = np.array(reward_history)
    
    if reversal_trial is not None:
        phases = ['Pre-reversal', 'Post-reversal']
        slices = [slice(0, reversal_trial), slice(reversal_trial, None)]
    else:
        phases = ['All trials']
        slices = [slice(None)]
    
    # Left: Action counts
    ax1 = axes[0]
    x = np.arange(len(phases))
    width = 0.35
    
    left_counts = [np.sum(action_history[s] == 0) for s in slices]
    right_counts = [np.sum(action_history[s] == 1) for s in slices]
    
    bars1 = ax1.bar(x - width/2, left_counts, width, label='Left', color='#e74c3c', alpha=0.8)
    bars2 = ax1.bar(x + width/2, right_counts, width, label='Right', color='#3498db', alpha=0.8)
    
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Action Distribution', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases)
    ax1.legend()
    
    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Right: Reward rate per action
    ax2 = axes[1]
    
    left_rewards = []
    right_rewards = []
    
    for s in slices:
        left_mask = action_history[s] == 0
        right_mask = action_history[s] == 1
        
        left_r = np.mean(reward_history[s][left_mask]) if np.any(left_mask) else 0
        right_r = np.mean(reward_history[s][right_mask]) if np.any(right_mask) else 0
        
        left_rewards.append(left_r)
        right_rewards.append(right_r)
    
    bars3 = ax2.bar(x - width/2, left_rewards, width, label='Left', color='#e74c3c', alpha=0.8)
    bars4 = ax2.bar(x + width/2, right_rewards, width, label='Right', color='#3498db', alpha=0.8)
    
    ax2.set_ylabel('Reward Rate', fontsize=11)
    ax2.set_title('Reward Rate by Action', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(phases)
    ax2.set_ylim(0, 1)
    ax2.axhline(y=0.6, color='#27ae60', linestyle='--', alpha=0.7, label='Better option')
    ax2.axhline(y=0.4, color='#f39c12', linestyle='--', alpha=0.7, label='Worse option')
    ax2.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    return fig


def create_dashboard_figure(
    rpe_history: np.ndarray,
    reward_history: np.ndarray,
    action_history: np.ndarray,
    policy_history: np.ndarray,
    value_history: np.ndarray,
    reversal_trial: Optional[int] = None,
    current_trial: Optional[int] = None,
    decoded_intention: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a comprehensive dashboard showing all agent metrics.
    
    This is the main visualization for the real-time UI.
    
    Parameters
    ----------
    rpe_history : np.ndarray
        RPE values for each trial
    reward_history : np.ndarray
        Rewards received
    action_history : np.ndarray
        Actions taken
    policy_history : np.ndarray
        Policy probabilities over time
    value_history : np.ndarray
        Value estimates over time
    reversal_trial : int, optional
        Trial where reversal occurred
    current_trial : int, optional
        Current trial number (for live display)
    decoded_intention : str, optional
        Current BCI decoded intention ('Left' or 'Right')
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Convert to numpy arrays
    rpe_history = np.array(rpe_history)
    reward_history = np.array(reward_history)
    action_history = np.array(action_history)
    policy_history = np.array(policy_history)
    value_history = np.array(value_history)
    
    trials = np.arange(len(rpe_history))
    
    # 1. Current state indicator (top-left)
    ax_status = fig.add_subplot(gs[0, 0])
    ax_status.axis('off')
    
    trial_text = f"Trial: {current_trial}" if current_trial else f"Trial: {len(rpe_history)}"
    intention_text = f"Decoded: {decoded_intention}" if decoded_intention else "Decoded: N/A"
    
    if len(policy_history) > 0:
        current_policy = policy_history[-1]
        policy_text = f"Policy: L={current_policy[0]:.1%} | R={current_policy[1]:.1%}"
    else:
        policy_text = "Policy: N/A"
    
    last_rpe = rpe_history[-1] if len(rpe_history) > 0 else 0
    rpe_color = '#27ae60' if last_rpe > 0 else '#e74c3c'
    
    status_text = f"""
    🧠 BRAIN-BEHAVIOR DASHBOARD
    {'='*35}
    
    {trial_text}
    {intention_text}
    {policy_text}
    
    Last RPE: {last_rpe:+.3f}
    Total Reward: {np.sum(reward_history)}/{len(reward_history)}
    """
    
    ax_status.text(0.1, 0.9, status_text, transform=ax_status.transAxes, fontsize=11,
                  verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. RPE over time (top-center + top-right)
    ax_rpe = fig.add_subplot(gs[0, 1:])
    colors = ['#e74c3c' if r < 0 else '#27ae60' for r in rpe_history]
    ax_rpe.bar(trials, rpe_history, color=colors, alpha=0.6, width=1.0)
    ax_rpe.axhline(y=0, color='black', linestyle='-', linewidth=1)
    if reversal_trial:
        ax_rpe.axvline(x=reversal_trial, color='#9b59b6', linestyle='--', linewidth=2)
    ax_rpe.set_ylabel('RPE (δ)')
    ax_rpe.set_title('⚡ Reward Prediction Error (Dopamine Signal)', fontweight='bold')
    
    # 3. Policy evolution (middle-left)
    ax_policy = fig.add_subplot(gs[1, 0])
    if len(policy_history) > 0:
        ax_policy.plot(trials, policy_history[:, 0], 'r-', label='P(Left)', linewidth=2)
        ax_policy.plot(trials, policy_history[:, 1], 'b-', label='P(Right)', linewidth=2)
    ax_policy.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    if reversal_trial:
        ax_policy.axvline(x=reversal_trial, color='#9b59b6', linestyle='--', linewidth=2)
    ax_policy.set_ylim(0, 1)
    ax_policy.set_ylabel('Probability')
    ax_policy.set_title('🎯 Policy Evolution')
    ax_policy.legend(loc='best', fontsize=8)
    
    # 4. Cumulative reward (middle-center)
    ax_reward = fig.add_subplot(gs[1, 1])
    cumulative = np.cumsum(reward_history)
    ax_reward.fill_between(trials, 0, cumulative, alpha=0.3, color='#27ae60')
    ax_reward.plot(trials, cumulative, color='#27ae60', linewidth=2)
    ax_reward.plot(trials, trials * 0.6, 'k--', alpha=0.5, label='Optimal')
    if reversal_trial:
        ax_reward.axvline(x=reversal_trial, color='#9b59b6', linestyle='--', linewidth=2)
    ax_reward.set_ylabel('Cumulative Reward')
    ax_reward.set_title('💰 Total Reward')
    ax_reward.legend(loc='upper left', fontsize=8)
    
    # 5. Value estimates (middle-right)
    ax_value = fig.add_subplot(gs[1, 2])
    ax_value.plot(trials, value_history, color='#9b59b6', linewidth=2)
    if reversal_trial:
        ax_value.axvline(x=reversal_trial, color='black', linestyle='--', linewidth=2)
    ax_value.set_ylabel('V(s)')
    ax_value.set_title('📊 Value Estimate')
    
    # 6. Action history (bottom-left)
    ax_actions = fig.add_subplot(gs[2, 0])
    window = 20
    if len(action_history) >= window:
        rolling_right = np.convolve(action_history, np.ones(window)/window, mode='valid')
        x_roll = np.arange(window-1, len(action_history))
        ax_actions.fill_between(x_roll, 0.5, rolling_right, 
                               where=(rolling_right > 0.5), alpha=0.3, color='#3498db')
        ax_actions.fill_between(x_roll, rolling_right, 0.5, 
                               where=(rolling_right < 0.5), alpha=0.3, color='#e74c3c')
        ax_actions.plot(x_roll, rolling_right, color='black', linewidth=1)
    ax_actions.axhline(y=0.5, color='gray', linestyle='--')
    if reversal_trial:
        ax_actions.axvline(x=reversal_trial, color='#9b59b6', linestyle='--', linewidth=2)
    ax_actions.set_ylim(0, 1)
    ax_actions.set_xlabel('Trial')
    ax_actions.set_ylabel('P(Right)')
    ax_actions.set_title('🎮 Choice Behavior')
    
    # 7. Reward rate (bottom-center)
    ax_rate = fig.add_subplot(gs[2, 1])
    if len(reward_history) >= window:
        rolling_reward = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        ax_rate.plot(x_roll, rolling_reward, color='#f39c12', linewidth=2)
        ax_rate.axhline(y=0.6, color='#27ae60', linestyle='--', alpha=0.7)
    ax_rate.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    if reversal_trial:
        ax_rate.axvline(x=reversal_trial, color='#9b59b6', linestyle='--', linewidth=2)
    ax_rate.set_ylim(0, 1)
    ax_rate.set_xlabel('Trial')
    ax_rate.set_ylabel('Reward Rate')
    ax_rate.set_title('📈 Success Rate')
    
    # 8. Legend / Info (bottom-right)
    ax_info = fig.add_subplot(gs[2, 2])
    ax_info.axis('off')
    
    # Calculate summary stats
    total_reward = np.sum(reward_history)
    reward_rate = np.mean(reward_history) if len(reward_history) > 0 else 0
    mean_rpe = np.mean(rpe_history) if len(rpe_history) > 0 else 0
    
    info_text = f"""
    📋 SESSION SUMMARY
    {'='*25}
    
    Total Trials: {len(rpe_history)}
    Total Reward: {total_reward}
    Reward Rate: {reward_rate:.1%}
    Mean RPE: {mean_rpe:+.4f}
    
    LEGEND:
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    🟢 Positive RPE (reward > expected)
    🔴 Negative RPE (reward < expected)
    🟣 Reversal point
    """
    
    ax_info.text(0.1, 0.95, info_text, transform=ax_info.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('🧠 Brain–Behavior Mapping: Neuro-Inspired RL Dashboard', 
                fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Dashboard saved to {save_path}")
    
    return fig


# Convenience function for quick plotting
def quick_plot(agent_stats: Dict, reversal_trial: int = 500) -> None:
    """
    Quick visualization of agent statistics.
    
    Parameters
    ----------
    agent_stats : dict
        Output from ActorCriticAgent.get_statistics()
    reversal_trial : int
        Trial where reversal occurred
    """
    plot_rpe_dynamics(agent_stats['rpe_history'], reversal_trial)
    plt.show()
    
    plot_reward_history(agent_stats['reward_history'], 
                       agent_stats['action_history'], 
                       reversal_trial)
    plt.show()
    
    plot_policy_evolution(agent_stats['policy_history'], reversal_trial)
    plt.show()


if __name__ == "__main__":
    # Test with synthetic data
    print("🧪 Testing visualization utilities...")
    
    np.random.seed(42)
    n_trials = 1000
    reversal = 500
    
    # Generate fake data
    rpe = np.random.randn(n_trials) * 0.5
    rpe[reversal:reversal+50] += np.linspace(0, -1, 50)  # Simulate reversal spike
    
    rewards = np.random.binomial(1, 0.6, n_trials)
    rewards[reversal:] = np.random.binomial(1, 0.4, n_trials - reversal)
    
    actions = np.random.binomial(1, 0.6, n_trials)
    
    policy = np.column_stack([
        np.linspace(0.4, 0.3, n_trials),
        np.linspace(0.6, 0.7, n_trials)
    ])
    
    values = np.cumsum(rpe) * 0.01
    
    print("✅ Creating dashboard...")
    fig = create_dashboard_figure(rpe, rewards, actions, policy, values, reversal)
    plt.show()
    
    print("✅ All visualization tests passed!")
