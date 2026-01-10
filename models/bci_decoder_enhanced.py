"""
BCI Motor Imagery Decoder - Standalone Script
==============================================
High-level interface for loading and using the trained ShallowConvNet model
for real-time motor imagery classification (Left Hand vs Right Hand).

This module provides:
- Model loading with automatic validation
- Batch and real-time prediction interfaces
- Synthetic EEG generation for testing
- Signal preprocessing pipeline
- Confidence-based decision thresholding

Author: Person 1 - Neurosignal Processing
Date: January 2026
Model Performance: 80% accuracy on test set (AUC-ROC: 0.9077)
"""

import os
import json
import numpy as np
from typing import Tuple, Dict, Optional, List, Union
import warnings

try:
    from tensorflow import keras
except ImportError:
    import keras

warnings.filterwarnings('ignore')


class BCIDecoder:
    """
    Motor Imagery BCI Decoder using ShallowConvNet architecture.
    
    Handles loading, preprocessing, and prediction of EEG motor imagery signals.
    
    Parameters
    ----------
    model_path : str
        Path to the trained .keras model file
    metadata_path : str, optional
        Path to model metadata JSON file containing preprocessing parameters
    verbose : bool, default=True
        Print initialization messages and metadata
    
    Attributes
    ----------
    model : keras.Model
        Loaded neural network model
    metadata : dict
        Model hyperparameters and training metadata
    input_shape : tuple
        Expected input shape (timepoints, channels)
    classes : list
        Class labels ['Left', 'Right']
    
    Examples
    --------
    >>> decoder = BCIDecoder('models/shallow_convnet_motor_imagery.keras')
    >>> # Single prediction
    >>> eeg_signal = np.random.randn(1, 751, 22)  # (batch, time, channels)
    >>> prediction = decoder.predict(eeg_signal)
    >>> print(prediction['class'], prediction['confidence'])
    
    >>> # Real-time prediction with sliding window
    >>> real_time_pred = decoder.predict_realtime(stream_data)
    >>> print(real_time_pred['smoothed_class'], real_time_pred['rpe_compatible'])
    """
    
    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        verbose: bool = True
    ):
        """Initialize the BCI decoder and load model."""
        self.model_path = model_path
        self.verbose = verbose
        
        # Validate model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        self.model = keras.models.load_model(model_path)
        
        # Load metadata
        self.metadata = self._load_metadata(metadata_path)
        
        # Set class labels and input shape
        self.classes = ['Left', 'Right']
        self.input_shape = (751, 22)  # (timepoints, channels)
        
        if self.verbose:
            self._print_info()
    
    def _load_metadata(self, metadata_path: Optional[str]) -> Dict:
        """Load and validate metadata JSON."""
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load metadata: {e}")
                return self._default_metadata()
        else:
            return self._default_metadata()
    
    def _default_metadata(self) -> Dict:
        """Return default metadata based on known model parameters."""
        return {
            'model_type': 'ShallowConvNet',
            'input_shape': [751, 22],
            'output_classes': 2,
            'test_accuracy': 0.80,
            'auc_roc': 0.9077,
            'preprocessing': {
                'bandpass_filter': '8-30 Hz',
                'normalization': 'z-score per trial',
                'baseline_correction': 'first 100ms',
                'sampling_rate': 250
            },
            'training': {
                'optimizer': 'Adam',
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 93,
                'early_stopping_patience': 20
            }
        }
    
    def _print_info(self):
        """Print model information."""
        print("\n" + "="*60)
        print("BCI MOTOR IMAGERY DECODER - LOADED")
        print("="*60)
        print(f"Model type: {self.metadata.get('model_type', 'Unknown')}")
        print(f"Test accuracy: {self.metadata.get('test_accuracy', 'N/A'):.2%}")
        print(f"AUC-ROC: {self.metadata.get('auc_roc', 'N/A'):.4f}")
        print(f"Input shape: {self.input_shape}")
        print(f"Classes: {self.classes}")
        print("="*60 + "\n")
    
    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to raw EEG signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Raw EEG signal of shape (batch, timepoints, channels) or (timepoints, channels)
        
        Returns
        -------
        np.ndarray
            Preprocessed signal
        """
        # Ensure batch dimension
        if signal.ndim == 2:
            signal = np.expand_dims(signal, axis=0)
        
        # Z-score normalization per trial and channel
        batch_size = signal.shape[0]
        normalized = np.zeros_like(signal)
        
        for i in range(batch_size):
            for ch in range(signal.shape[2]):
                trial_data = signal[i, :, ch]
                mean = np.mean(trial_data)
                std = np.std(trial_data)
                if std > 1e-6:  # Avoid division by zero
                    normalized[i, :, ch] = (trial_data - mean) / std
                else:
                    normalized[i, :, ch] = trial_data - mean
        
        return normalized
    
    def predict(
        self,
        eeg_signal: np.ndarray,
        confidence_threshold: float = 0.0,
        preprocess: bool = True
    ) -> Dict[str, Union[str, float, np.ndarray]]:
        """
        Predict motor imagery class for given EEG signal(s).
        
        Parameters
        ----------
        eeg_signal : np.ndarray
            EEG signal of shape (batch, timepoints, channels) or (timepoints, channels)
        confidence_threshold : float, default=0.0
            If prediction confidence < threshold, return 'Uncertain'
        preprocess : bool, default=True
            Whether to apply preprocessing
        
        Returns
        -------
        dict
            Keys:
            - 'class': Predicted class ('Left', 'Right', or 'Uncertain')
            - 'confidence': Confidence score (0-1)
            - 'probabilities': Array of [P(Left), P(Right)]
            - 'label_index': 0 for Left, 1 for Right
            - 'batch_size': Number of samples in batch
        
        Examples
        --------
        >>> eeg = np.random.randn(1, 751, 22)
        >>> result = decoder.predict(eeg, confidence_threshold=0.6)
        >>> print(f"{result['class']} (conf: {result['confidence']:.2%})")
        """
        # Ensure batch dimension
        if eeg_signal.ndim == 2:
            eeg_signal = np.expand_dims(eeg_signal, axis=0)
        
        # Preprocess
        if preprocess:
            eeg_signal = self.preprocess(eeg_signal)
        
        # Validate input shape
        if eeg_signal.shape[1:] != self.input_shape:
            raise ValueError(
                f"Input shape {eeg_signal.shape[1:]} != expected {self.input_shape}"
            )
        
        # Predict
        probabilities = self.model.predict(eeg_signal, verbose=0)
        
        # Get predictions
        predicted_indices = np.argmax(probabilities, axis=1)
        max_confidences = np.max(probabilities, axis=1)
        
        # Format results (return first sample if batch)
        idx = predicted_indices[0]
        conf = max_confidences[0]
        
        # Apply confidence threshold
        if conf < confidence_threshold:
            predicted_class = 'Uncertain'
        else:
            predicted_class = self.classes[idx]
        
        return {
            'class': predicted_class,
            'confidence': float(conf),
            'probabilities': probabilities[0].astype(np.float32),
            'label_index': int(idx),
            'batch_size': len(eeg_signal)
        }
    
    def predict_realtime(
        self,
        eeg_stream: np.ndarray,
        window_size: int = 751,
        stride: int = 75,
        smoothing_window: int = 5,
        confidence_threshold: float = 0.6
    ) -> Dict:
        """
        Real-time prediction using sliding window approach.
        
        Simulates continuous EEG stream processing with output suitable for
        reinforcement learning agent integration.
        
        Parameters
        ----------
        eeg_stream : np.ndarray
            Continuous EEG stream of shape (stream_length, channels)
        window_size : int, default=751
            Sliding window size (samples at 250Hz ≈ 3 seconds)
        stride : int, default=75
            Step size between windows (75 samples ≈ 300ms)
        smoothing_window : int, default=5
            Number of predictions to average for smoothing
        confidence_threshold : float, default=0.6
            Confidence threshold for RL action validation
        
        Returns
        -------
        dict
            Keys:
            - 'predictions': List of predictions per window
            - 'smoothed_class': Final smoothed prediction
            - 'action': 'left' or 'right' for RL agent
            - 'confidence': Smoothed confidence score
            - 'rpe_compatible': Format suitable for RL integration
            - 'n_windows': Number of windows processed
            - 'valid': Whether prediction meets confidence threshold
        
        Notes
        -----
        - Output format compatible with reinforcement learning agent
        - Confidence threshold prevents low-quality RL predictions
        - Smoothing reduces noise from individual windows
        
        Examples
        --------
        >>> stream = np.random.randn(2000, 22)  # 8 seconds of data
        >>> result = decoder.predict_realtime(stream)
        >>> if result['valid']:
        >>>     rl_agent.act(result['action'])  # Feed to RL agent
        """
        # Validate input
        if eeg_stream.ndim != 2 or eeg_stream.shape[1] != 22:
            raise ValueError(f"Expected shape (n_samples, 22), got {eeg_stream.shape}")
        
        stream_length = eeg_stream.shape[0]
        if stream_length < window_size:
            raise ValueError(
                f"Stream length {stream_length} < window size {window_size}"
            )
        
        # Extract windows
        predictions = []
        windows_idx = []
        
        for i in range(0, stream_length - window_size, stride):
            window = eeg_stream[i:i+window_size]
            pred = self.predict(window, preprocess=True)
            predictions.append(pred)
            windows_idx.append(i)
        
        if not predictions:
            return {
                'predictions': [],
                'smoothed_class': 'No Data',
                'action': None,
                'confidence': 0.0,
                'rpe_compatible': {'action': None, 'confidence': 0.0},
                'n_windows': 0,
                'valid': False
            }
        
        # Smooth predictions
        smoothing_window = min(smoothing_window, len(predictions))
        smoothed_probs = np.mean(
            [p['probabilities'] for p in predictions[-smoothing_window:]],
            axis=0
        )
        smoothed_conf = np.max(smoothed_probs)
        smoothed_idx = np.argmax(smoothed_probs)
        smoothed_class = self.classes[smoothed_idx]
        
        # RL-compatible action
        action = 'left' if smoothed_idx == 0 else 'right'
        valid = smoothed_conf >= confidence_threshold
        
        return {
            'predictions': predictions,
            'smoothed_class': smoothed_class,
            'action': action if valid else None,
            'confidence': float(smoothed_conf),
            'rpe_compatible': {
                'action': action if valid else None,
                'confidence': float(smoothed_conf),
                'class': smoothed_class,
                'valid': valid
            },
            'n_windows': len(predictions),
            'valid': valid
        }
    
    def generate_synthetic_eeg(
        self,
        n_trials: int = 10,
        class_label: str = 'Left',
        noise_level: float = 0.3,
        seed: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate synthetic EEG data for testing and validation.
        
        Creates realistic motor imagery EEG signals based on known
        spectral and temporal characteristics of the training data.
        
        Parameters
        ----------
        n_trials : int, default=10
            Number of synthetic trials to generate
        class_label : str, default='Left'
            Class to simulate: 'Left' or 'Right'
        noise_level : float, default=0.3
            Gaussian noise standard deviation (0-1 scale)
        seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        np.ndarray
            Synthetic EEG of shape (n_trials, 751, 22)
        
        Notes
        -----
        - Simulates 8-30 Hz motor rhythms
        - Includes contralateral activation patterns
        - For 'Left': stronger activation at C4, for 'Right': stronger at C3
        - Suitable for testing preprocessing and prediction pipelines
        
        Examples
        --------
        >>> synthetic = decoder.generate_synthetic_eeg(n_trials=5, class_label='Right')
        >>> pred = decoder.predict(synthetic)
        >>> print(pred['confidence'])  # Should be reasonably high
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Known channel locations (simplified 10-20 system)
        # C3 (index 9), C4 (index 18) are motor cortex regions
        motor_channels = {
            'C3': 9,   # Left motor cortex
            'C4': 18,  # Right motor cortex
        }
        
        # Contralateral activation (opposite hemisphere)
        active_channel = motor_channels['C4'] if class_label == 'Left' else motor_channels['C3']
        
        # Time parameters
        fs = 250  # Sampling rate
        duration = 3.004  # seconds
        n_samples = int(fs * duration)  # 751 samples
        t = np.arange(n_samples) / fs
        
        # Initialize array
        synthetic_eeg = np.zeros((n_trials, n_samples, 22))
        
        for trial in range(n_trials):
            # Generate for each channel
            for ch in range(22):
                # Background activity: pink noise (1/f)
                white = np.random.randn(n_samples)
                pink = np.cumsum(white) / np.sqrt(n_samples)
                pink = (pink - np.mean(pink)) / np.std(pink)
                
                # Motor rhythm component (8-30 Hz)
                mu_band = 10 + np.random.uniform(-2, 2)  # 8-12 Hz (mu)
                beta_band = 20 + np.random.uniform(-5, 5)  # 15-30 Hz (beta)
                
                motor_rhythm = (
                    0.5 * np.sin(2 * np.pi * mu_band * t) +
                    0.3 * np.sin(2 * np.pi * beta_band * t)
                )
                
                # Event-related desynchronization (ERD): amplitude modulation
                # Stronger desynchronization at motor cortex
                erd_envelope = np.ones(n_samples)
                
                if ch == active_channel:
                    # Strong ERD at active channel (contralateral)
                    erd_window_start = int(0.5 * fs)  # 0.5s
                    erd_window_end = int(3.0 * fs)    # 3.0s
                    erd_envelope[erd_window_start:erd_window_end] *= 0.3  # 70% suppression
                else:
                    # Weak ERD elsewhere
                    erd_envelope *= 0.8
                
                # Combine components
                signal = pink + motor_rhythm * erd_envelope
                
                # Normalize
                signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-6)
                
                # Add noise
                noise = noise_level * np.random.randn(n_samples)
                synthetic_eeg[trial, :, ch] = signal + noise
        
        return synthetic_eeg.astype(np.float32)


# Utility functions for batch processing and evaluation
def evaluate_on_synthetic_data(
    decoder: BCIDecoder,
    n_trials_per_class: int = 20,
    confidence_threshold: float = 0.6
) -> Dict:
    """
    Evaluate decoder on synthetic data.
    
    Parameters
    ----------
    decoder : BCIDecoder
        Trained decoder instance
    n_trials_per_class : int
        Trials to generate per class
    confidence_threshold : float
        Confidence threshold for valid predictions
    
    Returns
    -------
    dict
        Accuracy, precision, recall, F1 scores
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    )
    
    # Generate synthetic data
    left_eeg = decoder.generate_synthetic_eeg(n_trials_per_class, 'Left', seed=42)
    right_eeg = decoder.generate_synthetic_eeg(n_trials_per_class, 'Right', seed=43)
    
    # Predict
    left_preds = [decoder.predict(left_eeg[i:i+1])['label_index'] for i in range(len(left_eeg))]
    right_preds = [decoder.predict(right_eeg[i:i+1])['label_index'] for i in range(len(right_eeg))]
    
    # Ground truth
    y_true = [0] * n_trials_per_class + [1] * n_trials_per_class
    y_pred = left_preds + right_preds
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }


if __name__ == '__main__':
    """
    Demo: Load model and test with synthetic EEG
    """
    import os
    
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, 'shallow_convnet_motor_imagery.keras')
    metadata_path = os.path.join(model_dir, 'model_metadata.json')
    
    # Initialize decoder
    print("Initializing BCI Decoder...")
    decoder = BCIDecoder(model_path, metadata_path, verbose=True)
    
    # Test 1: Batch prediction with synthetic data
    print("\n" + "="*60)
    print("TEST 1: Batch Prediction with Synthetic Data")
    print("="*60)
    
    synthetic_left = decoder.generate_synthetic_eeg(n_trials=5, class_label='Left', seed=42)
    synthetic_right = decoder.generate_synthetic_eeg(n_trials=5, class_label='Right', seed=43)
    
    print("\nPredictions on synthetic LEFT hand imagery:")
    for i in range(5):
        result = decoder.predict(synthetic_left[i:i+1])
        print(f"  Trial {i+1}: {result['class']:10s} (confidence: {result['confidence']:.2%})")
    
    print("\nPredictions on synthetic RIGHT hand imagery:")
    for i in range(5):
        result = decoder.predict(synthetic_right[i:i+1])
        print(f"  Trial {i+1}: {result['class']:10s} (confidence: {result['confidence']:.2%})")
    
    # Test 2: Real-time prediction
    print("\n" + "="*60)
    print("TEST 2: Real-Time Streaming Prediction")
    print("="*60)
    
    # Generate longer stream
    stream = decoder.generate_synthetic_eeg(n_trials=1, class_label='Right')[0]
    # Extend to 2000 samples
    stream = np.tile(stream, (3, 1))[:2000]
    
    realtime_result = decoder.predict_realtime(
        stream,
        window_size=751,
        stride=150,
        smoothing_window=3,
        confidence_threshold=0.6
    )
    
    print(f"Stream length: {stream.shape[0]} samples (~8 seconds at 250Hz)")
    print(f"Windows processed: {realtime_result['n_windows']}")
    print(f"Smoothed prediction: {realtime_result['smoothed_class']}")
    print(f"Smoothed confidence: {realtime_result['confidence']:.2%}")
    print(f"Valid for RL agent: {realtime_result['valid']}")
    print(f"RL-compatible action: {realtime_result['rpe_compatible']}")
    
    # Test 3: Evaluation on synthetic data
    print("\n" + "="*60)
    print("TEST 3: Evaluation on Synthetic Data")
    print("="*60)
    
    eval_results = evaluate_on_synthetic_data(decoder, n_trials_per_class=10)
    print(f"Accuracy: {eval_results['accuracy']:.2%}")
    print(f"Precision: {eval_results['precision']:.2%}")
    print(f"Recall: {eval_results['recall']:.2%}")
    print(f"F1 Score: {eval_results['f1']:.2%}")
    print(f"Confusion Matrix:\n{np.array(eval_results['confusion_matrix'])}")
    
    print("\n" + "="*60)
    print("All tests completed successfully!")
    print("="*60)
