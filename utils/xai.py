"""
Explainability (XAI) Utilities for BCI Model
============================================
Person 3 Deliverable: SHAP / Integrated Gradients for EEG channel importance

This module provides tools to understand which EEG channels and time windows
are most important for the BCI decoder's predictions.

Integrates with:
- SHAP: For model-agnostic feature importance
- MNE: For EEG topographic visualization
- MOABB: For dataset information and channel montages

Author: Person 3 - Integration, UX & Explainability
Date: January 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict
import warnings

# Try importing XAI and EEG libraries
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

try:
    import mne
    from mne.viz import plot_topomap
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    warnings.warn("MNE not available. Install with: pip install mne")

try:
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    MOABB_AVAILABLE = True
except ImportError:
    MOABB_AVAILABLE = False
    warnings.warn("MOABB not available. Install with: pip install moabb")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


# Standard 10-20 EEG channel names for motor imagery (22 channels)
EEG_CHANNELS = [
    'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
    'P1', 'Pz', 'P2', 'POz'
]

# Motor cortex channels (most important for motor imagery)
MOTOR_CHANNELS = ['C3', 'C1', 'Cz', 'C2', 'C4', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4']


def compute_gradient_importance(
    model,
    eeg_data: np.ndarray,
    target_class: int = 1
) -> np.ndarray:
    """
    Compute gradient-based importance for EEG channels.
    
    Uses integrated gradients to determine which channels and time points
    contribute most to the model's prediction.
    
    Parameters
    ----------
    model : keras.Model
        Trained BCI decoder model
    eeg_data : np.ndarray
        EEG data of shape (n_samples, timepoints, channels) or (timepoints, channels)
    target_class : int
        Class to compute gradients for (0=Left, 1=Right)
    
    Returns
    -------
    importance : np.ndarray
        Importance scores of shape (timepoints, channels)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required for gradient computation")
    
    # Ensure batch dimension
    if eeg_data.ndim == 2:
        eeg_data = eeg_data[np.newaxis, ...]
    
    eeg_tensor = tf.constant(eeg_data, dtype=tf.float32)
    
    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch(eeg_tensor)
        predictions = model(eeg_tensor)
        target_output = predictions[:, target_class]
    
    gradients = tape.gradient(target_output, eeg_tensor)
    
    # Compute importance as gradient * input (saliency)
    importance = np.abs(gradients.numpy() * eeg_data)
    
    # Average over samples if multiple
    if importance.shape[0] > 1:
        importance = np.mean(importance, axis=0)
    else:
        importance = importance[0]
    
    return importance


def compute_integrated_gradients(
    model,
    eeg_data: np.ndarray,
    target_class: int = 1,
    baseline: Optional[np.ndarray] = None,
    n_steps: int = 50
) -> np.ndarray:
    """
    Compute integrated gradients for more accurate attribution.
    
    Integrated Gradients provides a more theoretically grounded
    attribution method that satisfies axioms like sensitivity and
    implementation invariance.
    
    Parameters
    ----------
    model : keras.Model
        Trained BCI decoder model
    eeg_data : np.ndarray
        EEG data of shape (timepoints, channels)
    target_class : int
        Class to compute attributions for
    baseline : np.ndarray, optional
        Baseline input (default: zeros)
    n_steps : int
        Number of interpolation steps
    
    Returns
    -------
    attributions : np.ndarray
        Attribution scores of shape (timepoints, channels)
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required for integrated gradients")
    
    # Ensure correct shape
    if eeg_data.ndim == 2:
        eeg_data = eeg_data[np.newaxis, ...]
    
    if baseline is None:
        baseline = np.zeros_like(eeg_data)
    elif baseline.ndim == 2:
        baseline = baseline[np.newaxis, ...]
    
    # Create interpolated inputs
    alphas = np.linspace(0, 1, n_steps)
    interpolated_inputs = np.array([
        baseline + alpha * (eeg_data - baseline) 
        for alpha in alphas
    ])
    interpolated_inputs = interpolated_inputs.squeeze(axis=1)
    
    # Compute gradients for all interpolated inputs
    interpolated_tensor = tf.constant(interpolated_inputs, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated_tensor)
        predictions = model(interpolated_tensor)
        target_output = predictions[:, target_class]
    
    gradients = tape.gradient(target_output, interpolated_tensor).numpy()
    
    # Integrate gradients
    avg_gradients = np.mean(gradients, axis=0)
    integrated_gradients = (eeg_data[0] - baseline[0]) * avg_gradients
    
    return integrated_gradients


def plot_channel_importance(
    importance: np.ndarray,
    channel_names: Optional[List[str]] = None,
    title: str = "EEG Channel Importance",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot importance scores per EEG channel.
    
    Parameters
    ----------
    importance : np.ndarray
        Importance scores of shape (timepoints, channels) or (channels,)
    channel_names : list, optional
        Names of channels (default: standard 22 motor imagery channels)
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    if channel_names is None:
        channel_names = EEG_CHANNELS[:importance.shape[-1]]
    
    # Aggregate over time if needed
    if importance.ndim == 2:
        channel_importance = np.mean(np.abs(importance), axis=0)
    else:
        channel_importance = np.abs(importance)
    
    # Normalize
    channel_importance = channel_importance / np.max(channel_importance)
    
    # Sort by importance
    sorted_idx = np.argsort(channel_importance)[::-1]
    sorted_channels = [channel_names[i] for i in sorted_idx]
    sorted_importance = channel_importance[sorted_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color bars by motor cortex membership
    colors = ['#e74c3c' if ch in MOTOR_CHANNELS else '#3498db' 
              for ch in sorted_channels]
    
    bars = ax.barh(range(len(sorted_channels)), sorted_importance, color=colors, alpha=0.8)
    
    ax.set_yticks(range(len(sorted_channels)))
    ax.set_yticklabels(sorted_channels)
    ax.set_xlabel('Normalized Importance', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Motor Cortex'),
        Patch(facecolor='#3498db', label='Other')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    # Add annotation
    ax.annotate(
        'Motor cortex channels (C3, C4, etc.)\nare expected to be most important\nfor motor imagery classification',
        xy=(0.95, 0.05), xycoords='axes fraction',
        fontsize=9, ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    return fig


def plot_temporal_importance(
    importance: np.ndarray,
    sampling_rate: int = 250,
    time_offset: float = 0.5,
    channel_names: Optional[List[str]] = None,
    title: str = "Temporal Importance Pattern",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot importance over time, showing when decisions are made.
    
    Parameters
    ----------
    importance : np.ndarray
        Importance scores of shape (timepoints, channels)
    sampling_rate : int
        EEG sampling rate in Hz
    time_offset : float
        Time offset from cue onset in seconds
    channel_names : list, optional
        Names of channels
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    if channel_names is None:
        channel_names = EEG_CHANNELS[:importance.shape[-1]]
    
    n_timepoints, n_channels = importance.shape
    
    # Create time axis
    time = np.linspace(time_offset, time_offset + n_timepoints/sampling_rate, n_timepoints)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Top: Heatmap of all channels over time
    ax1 = axes[0]
    im = ax1.imshow(np.abs(importance).T, aspect='auto', cmap='hot',
                   extent=[time[0], time[-1], 0, n_channels])
    ax1.set_yticks(np.arange(n_channels) + 0.5)
    ax1.set_yticklabels(channel_names, fontsize=8)
    ax1.set_ylabel('Channel')
    ax1.set_title(title, fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Importance')
    
    # Bottom: Average importance over time
    ax2 = axes[1]
    avg_importance = np.mean(np.abs(importance), axis=1)
    ax2.fill_between(time, 0, avg_importance, alpha=0.5, color='#3498db')
    ax2.plot(time, avg_importance, color='#3498db', linewidth=2)
    
    # Mark peak
    peak_idx = np.argmax(avg_importance)
    peak_time = time[peak_idx]
    ax2.axvline(x=peak_time, color='#e74c3c', linestyle='--', linewidth=2)
    ax2.annotate(f'Peak: {peak_time:.2f}s', xy=(peak_time, avg_importance[peak_idx]),
                xytext=(peak_time + 0.2, avg_importance[peak_idx] * 0.9),
                fontsize=10, color='#e74c3c',
                arrowprops=dict(arrowstyle='->', color='#e74c3c'))
    
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Average Importance', fontsize=11)
    
    # Add motor imagery timing annotations
    ax2.axvspan(1.0, 2.0, alpha=0.1, color='green', label='Expected MI peak (1-2s)')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    return fig


def plot_topographic_importance(
    channel_importance: np.ndarray,
    channel_names: Optional[List[str]] = None,
    title: str = "Topographic Channel Importance",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot channel importance on a topographic head map.
    
    Note: This is a simplified visualization. For accurate topomaps,
    use MNE-Python's plotting functions.
    
    Parameters
    ----------
    channel_importance : np.ndarray
        Importance scores per channel
    channel_names : list, optional
        Names of channels
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    if channel_names is None:
        channel_names = EEG_CHANNELS[:len(channel_importance)]
    
    # Simplified 2D positions for standard 10-20 system
    # These are approximate positions for visualization
    channel_positions = {
        'Fz': (0.5, 0.85), 'FC3': (0.3, 0.7), 'FC1': (0.4, 0.7),
        'FCz': (0.5, 0.7), 'FC2': (0.6, 0.7), 'FC4': (0.7, 0.7),
        'C5': (0.15, 0.5), 'C3': (0.3, 0.5), 'C1': (0.4, 0.5),
        'Cz': (0.5, 0.5), 'C2': (0.6, 0.5), 'C4': (0.7, 0.5), 'C6': (0.85, 0.5),
        'CP3': (0.3, 0.35), 'CP1': (0.4, 0.35), 'CPz': (0.5, 0.35),
        'CP2': (0.6, 0.35), 'CP4': (0.7, 0.35),
        'P1': (0.4, 0.2), 'Pz': (0.5, 0.2), 'P2': (0.6, 0.2),
        'POz': (0.5, 0.1)
    }
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw head outline
    head_circle = plt.Circle((0.5, 0.5), 0.45, fill=False, linewidth=2)
    ax.add_patch(head_circle)
    
    # Draw nose
    nose = plt.Polygon([(0.5, 0.95), (0.45, 0.9), (0.55, 0.9)], fill=False, linewidth=2)
    ax.add_patch(nose)
    
    # Draw ears
    left_ear = plt.Circle((0.02, 0.5), 0.03, fill=False, linewidth=2)
    right_ear = plt.Circle((0.98, 0.5), 0.03, fill=False, linewidth=2)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)
    
    # Normalize importance
    importance_norm = channel_importance / np.max(channel_importance)
    
    # Plot channels
    for i, (ch_name, importance) in enumerate(zip(channel_names, importance_norm)):
        if ch_name in channel_positions:
            x, y = channel_positions[ch_name]
            
            # Color by importance
            color = plt.cm.hot(importance)
            size = 200 + importance * 800
            
            ax.scatter(x, y, c=[color], s=size, zorder=3, alpha=0.8)
            ax.annotate(ch_name, (x, y), ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Normalized Importance')
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    
    # Add annotation
    ax.annotate(
        'Contralateral pattern expected:\nC3 for Right hand, C4 for Left hand',
        xy=(0.5, -0.05), ha='center', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to {save_path}")
    
    return fig


# =============================================================================
# SHAP-Based Explainability Functions
# =============================================================================

def compute_shap_values(
    model,
    eeg_data: np.ndarray,
    background_data: Optional[np.ndarray] = None,
    n_background: int = 50
) -> Optional[np.ndarray]:
    """
    Compute SHAP values for EEG channel importance using DeepExplainer.
    
    SHAP (SHapley Additive exPlanations) provides theoretically grounded
    feature attributions based on game theory.
    
    Parameters
    ----------
    model : keras.Model
        Trained BCI decoder model
    eeg_data : np.ndarray
        EEG data to explain, shape (n_samples, timepoints, channels)
    background_data : np.ndarray, optional
        Background samples for SHAP (default: random subset of eeg_data)
    n_background : int
        Number of background samples to use
    
    Returns
    -------
    shap_values : np.ndarray or None
        SHAP values for each input feature, or None if SHAP unavailable
    """
    if not SHAP_AVAILABLE:
        warnings.warn("SHAP not available. Returning None.")
        return None
    
    if not TF_AVAILABLE:
        warnings.warn("TensorFlow required for SHAP DeepExplainer.")
        return None
    
    # Ensure batch dimension
    if eeg_data.ndim == 2:
        eeg_data = eeg_data[np.newaxis, ...]
    
    # Create background data
    if background_data is None:
        if len(eeg_data) > n_background:
            idx = np.random.choice(len(eeg_data), n_background, replace=False)
            background_data = eeg_data[idx]
        else:
            background_data = eeg_data
    
    try:
        # Use DeepExplainer for deep learning models
        explainer = shap.DeepExplainer(model, background_data)
        shap_values = explainer.shap_values(eeg_data)
        
        # Handle list output (for multi-class)
        if isinstance(shap_values, list):
            # Return values for predicted class
            shap_values = np.array(shap_values)
        
        return shap_values
        
    except Exception as e:
        warnings.warn(f"SHAP computation failed: {e}. Using GradientExplainer fallback.")
        try:
            explainer = shap.GradientExplainer(model, background_data)
            shap_values = explainer.shap_values(eeg_data)
            return np.array(shap_values) if isinstance(shap_values, list) else shap_values
        except Exception as e2:
            warnings.warn(f"GradientExplainer also failed: {e2}")
            return None


def plot_shap_summary(
    shap_values: np.ndarray,
    channel_names: Optional[List[str]] = None,
    class_names: List[str] = ['Left', 'Right'],
    title: str = "SHAP Feature Importance Summary",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot SHAP summary showing channel importance across classes.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values from compute_shap_values()
    channel_names : list, optional
        Names of EEG channels
    class_names : list
        Names of output classes
    title : str
        Plot title
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.Figure
    """
    if channel_names is None:
        channel_names = EEG_CHANNELS
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    for class_idx, (ax, class_name) in enumerate(zip(axes, class_names)):
        if shap_values.ndim == 4:  # (n_classes, n_samples, timepoints, channels)
            class_shap = shap_values[class_idx]
        else:
            class_shap = shap_values
        
        # Average over time and samples
        channel_importance = np.mean(np.abs(class_shap), axis=(0, 1))
        
        if len(channel_importance) != len(channel_names):
            channel_names = [f'Ch{i}' for i in range(len(channel_importance))]
        
        # Sort by importance
        sorted_idx = np.argsort(channel_importance)[::-1]
        sorted_channels = [channel_names[i] for i in sorted_idx]
        sorted_importance = channel_importance[sorted_idx]
        
        # Color by motor cortex
        colors = ['#e74c3c' if ch in MOTOR_CHANNELS else '#3498db' 
                  for ch in sorted_channels]
        
        ax.barh(range(len(sorted_channels)), sorted_importance, color=colors, alpha=0.8)
        ax.set_yticks(range(len(sorted_channels)))
        ax.set_yticklabels(sorted_channels, fontsize=9)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f'{class_name} Hand Imagery', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
    
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ SHAP summary saved to {save_path}")
    
    return fig


# =============================================================================
# MNE-Based Topographic Visualization
# =============================================================================

def create_mne_info(
    channel_names: Optional[List[str]] = None,
    sfreq: float = 250.0
) -> Optional['mne.Info']:
    """
    Create MNE Info object for EEG channel configuration.
    
    Parameters
    ----------
    channel_names : list, optional
        Names of EEG channels (default: standard 22-channel motor imagery)
    sfreq : float
        Sampling frequency in Hz
    
    Returns
    -------
    info : mne.Info or None
        MNE Info object, or None if MNE unavailable
    """
    if not MNE_AVAILABLE:
        warnings.warn("MNE not available for topographic plotting.")
        return None
    
    if channel_names is None:
        channel_names = EEG_CHANNELS
    
    # Create MNE info with standard 10-20 montage
    info = mne.create_info(
        ch_names=channel_names,
        sfreq=sfreq,
        ch_types='eeg'
    )
    
    # Set standard montage
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage, on_missing='ignore')
    except Exception as e:
        warnings.warn(f"Could not set montage: {e}")
    
    return info


def plot_mne_topomap(
    channel_values: np.ndarray,
    channel_names: Optional[List[str]] = None,
    title: str = "EEG Topographic Map",
    cmap: str = 'RdBu_r',
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot EEG topographic map using MNE visualization.
    
    Creates a proper scalp topography showing spatial distribution
    of values across EEG channels.
    
    Parameters
    ----------
    channel_values : np.ndarray
        Values per channel (importance, amplitude, etc.)
    channel_names : list, optional
        Names of EEG channels
    title : str
        Plot title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    fig : matplotlib.Figure or None
        Figure object, or None if MNE unavailable
    """
    if not MNE_AVAILABLE:
        warnings.warn("MNE not available. Using fallback topographic plot.")
        return plot_topographic_importance(channel_values, channel_names, title, figsize, save_path)
    
    if channel_names is None:
        channel_names = EEG_CHANNELS[:len(channel_values)]
    
    # Create MNE info
    info = create_mne_info(channel_names)
    if info is None:
        return plot_topographic_importance(channel_values, channel_names, title, figsize, save_path)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        # Plot topomap
        im, _ = plot_topomap(
            channel_values,
            info,
            axes=ax,
            cmap=cmap,
            show=False,
            contours=6,
            sensors=True,
            names=channel_names
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Importance')
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        
        # Add motor cortex annotation
        ax.annotate(
            'Motor Cortex: C3 (Left) ↔ C4 (Right)\nContralateral activation expected',
            xy=(0.5, -0.1), xycoords='axes fraction',
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
    except Exception as e:
        warnings.warn(f"MNE topomap failed: {e}. Using fallback.")
        plt.close(fig)
        return plot_topographic_importance(channel_values, channel_names, title, figsize, save_path)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ MNE topomap saved to {save_path}")
    
    return fig


# =============================================================================
# MOABB Dataset Information
# =============================================================================

def get_dataset_info() -> Dict:
    """
    Get information about the BNCI2014-001 motor imagery dataset.
    
    Returns
    -------
    info : dict
        Dataset metadata and configuration
    """
    info = {
        'name': 'BNCI2014-001',
        'description': 'Four-class motor imagery dataset (we use 2 classes)',
        'n_subjects': 9,
        'n_sessions': 2,
        'n_runs': 6,
        'n_channels': 22,
        'sampling_rate': 250,
        'classes': {
            'Left': 'Left hand motor imagery',
            'Right': 'Right hand motor imagery'
        },
        'trial_duration': '4 seconds (cue at 2s, imagery 2-6s)',
        'channel_names': EEG_CHANNELS,
        'motor_channels': MOTOR_CHANNELS
    }
    
    if MOABB_AVAILABLE:
        try:
            dataset = BNCI2014_001()
            info['moabb_info'] = str(dataset)
        except Exception as e:
            info['moabb_error'] = str(e)
    else:
        info['moabb_available'] = False
    
    return info


def load_sample_eeg_data(
    subject_id: int = 1,
    n_trials: int = 10
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load sample EEG data from MOABB for testing and demonstration.
    
    Parameters
    ----------
    subject_id : int
        Subject ID (1-9)
    n_trials : int
        Number of trials to load
    
    Returns
    -------
    X : np.ndarray or None
        EEG data of shape (n_trials, timepoints, channels)
    y : np.ndarray or None
        Labels (0=Left, 1=Right)
    """
    if not MOABB_AVAILABLE:
        warnings.warn("MOABB not available. Cannot load real EEG data.")
        return None, None
    
    try:
        dataset = BNCI2014_001()
        paradigm = MotorImagery(
            events=['left_hand', 'right_hand'],
            n_classes=2,
            fmin=8, fmax=30,
            tmin=0.5, tmax=3.5
        )
        
        X, y, _ = paradigm.get_data(dataset, subjects=[subject_id])
        
        # Limit to n_trials
        if len(X) > n_trials:
            idx = np.random.choice(len(X), n_trials, replace=False)
            X = X[idx]
            y = y[idx]
        
        # Convert labels to 0/1
        y = (y == 'right_hand').astype(int)
        
        # Transpose to (trials, timepoints, channels)
        X = np.transpose(X, (0, 2, 1))
        
        return X, y
        
    except Exception as e:
        warnings.warn(f"Failed to load MOABB data: {e}")
        return None, None


# =============================================================================
# Comprehensive Explanation Function (Enhanced)
# =============================================================================

def explain_prediction(
    model,
    eeg_sample: np.ndarray,
    prediction: Dict,
    save_dir: Optional[str] = None,
    show: bool = True
) -> Dict:
    """
    Generate comprehensive explanation for a single prediction.
    
    Parameters
    ----------
    model : keras.Model
        Trained BCI decoder model
    eeg_sample : np.ndarray
        Single EEG sample of shape (timepoints, channels)
    prediction : dict
        Prediction result from BCIDecoder.predict()
    save_dir : str, optional
        Directory to save explanation figures
    show : bool
        Whether to display figures
    
    Returns
    -------
    explanation : dict
        Dictionary containing all explanation data and figures
    """
    from pathlib import Path
    
    print(f"\n{'='*60}")
    print("🔍 GENERATING PREDICTION EXPLANATION")
    print(f"{'='*60}")
    print(f"Predicted class: {prediction.get('class', 'Unknown')}")
    print(f"Confidence: {prediction.get('confidence', 0):.1%}")
    
    explanation = {
        'prediction': prediction,
        'figures': []
    }
    
    # Compute importance
    target_class = prediction.get('action', 1)
    
    print("\n📊 Computing gradient-based importance...")
    try:
        importance = compute_gradient_importance(model, eeg_sample, target_class)
        explanation['importance'] = importance
        
        # Channel importance
        channel_importance = np.mean(np.abs(importance), axis=0)
        explanation['channel_importance'] = channel_importance
        
        # Top channels
        top_idx = np.argsort(channel_importance)[::-1][:5]
        top_channels = [EEG_CHANNELS[i] for i in top_idx]
        print(f"Top 5 important channels: {top_channels}")
        
        # Generate plots
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        fig1 = plot_channel_importance(
            importance,
            title=f"Channel Importance for '{prediction.get('class')}' Prediction",
            save_path=str(save_dir / 'channel_importance.png') if save_dir else None
        )
        explanation['figures'].append(fig1)
        
        fig2 = plot_temporal_importance(
            importance,
            title=f"Temporal Pattern for '{prediction.get('class')}' Prediction",
            save_path=str(save_dir / 'temporal_importance.png') if save_dir else None
        )
        explanation['figures'].append(fig2)
        
        fig3 = plot_topographic_importance(
            channel_importance,
            title=f"Topographic Importance for '{prediction.get('class')}' Prediction",
            save_path=str(save_dir / 'topographic_importance.png') if save_dir else None
        )
        explanation['figures'].append(fig3)
        
        # Try MNE topomap if available
        if MNE_AVAILABLE:
            print("\n🧠 Generating MNE topomap...")
            try:
                fig4 = plot_mne_topomap(
                    channel_importance,
                    title=f"MNE Topomap: {prediction.get('class')} Prediction",
                    save_path=str(save_dir / 'mne_topomap.png') if save_dir else None
                )
                explanation['figures'].append(fig4)
                explanation['mne_available'] = True
            except Exception as e:
                print(f"⚠️ MNE topomap failed: {e}")
                explanation['mne_error'] = str(e)
        
        # Try SHAP if available
        if SHAP_AVAILABLE:
            print("\n📈 Computing SHAP values...")
            try:
                # Need background data for SHAP
                background = np.random.randn(20, *eeg_sample.shape).astype(np.float32)
                shap_values = compute_shap_values(
                    model, 
                    eeg_sample[np.newaxis, ...], 
                    background
                )
                if shap_values is not None:
                    explanation['shap_values'] = shap_values
                    
                    # SHAP summary plot
                    fig5 = plot_shap_summary(
                        shap_values,
                        feature_names=EEG_CHANNELS,
                        title=f"SHAP Summary: {prediction.get('class')} Prediction",
                        save_path=str(save_dir / 'shap_summary.png') if save_dir else None
                    )
                    if fig5:
                        explanation['figures'].append(fig5)
                    explanation['shap_available'] = True
            except Exception as e:
                print(f"⚠️ SHAP analysis failed: {e}")
                explanation['shap_error'] = str(e)
        
        if show:
            plt.show()
        
    except Exception as e:
        print(f"⚠️ Could not compute importance: {e}")
        explanation['error'] = str(e)
    
    # Report library availability
    explanation['libraries'] = {
        'shap': SHAP_AVAILABLE,
        'mne': MNE_AVAILABLE,
        'moabb': MOABB_AVAILABLE
    }
    
    print(f"\n{'='*60}")
    print("✅ Explanation complete!")
    print(f"   SHAP available: {SHAP_AVAILABLE}")
    print(f"   MNE available: {MNE_AVAILABLE}")
    print(f"   MOABB available: {MOABB_AVAILABLE}")
    
    return explanation


# Main test
if __name__ == "__main__":
    print("🧪 Testing XAI utilities...")
    
    # Generate synthetic importance data
    np.random.seed(42)
    n_timepoints = 751
    n_channels = 22
    
    # Simulate importance with motor cortex emphasis
    importance = np.random.rand(n_timepoints, n_channels) * 0.3
    
    # Add higher importance to motor channels (C3=7, C4=11)
    importance[:, 7] += 0.5  # C3
    importance[:, 11] += 0.5  # C4
    
    # Add temporal peak around 1-2 seconds (250-500 samples)
    importance[250:500, :] *= 2
    
    # Test plots
    fig1 = plot_channel_importance(importance, title="Test: Channel Importance")
    fig2 = plot_temporal_importance(importance, title="Test: Temporal Pattern")
    fig3 = plot_topographic_importance(
        np.mean(np.abs(importance), axis=0), 
        title="Test: Topographic Map"
    )
    
    plt.show()
    print("✅ XAI tests complete!")
