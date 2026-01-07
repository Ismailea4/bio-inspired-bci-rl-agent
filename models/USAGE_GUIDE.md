
# TRAINED CLASSIFIER MODEL - USAGE GUIDE
# =======================================

## Load Trained Model

```python
from tensorflow import keras
from pathlib import Path

# Load model
model = keras.models.load_model('models/shallow_convnet_motor_imagery.keras')

# Load metadata
import json
with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"Model: {metadata['model_name']}")
print(f"Task: {metadata['task']}")
print(f"Test Accuracy: {metadata['performance']['test_accuracy']:.4f}")
```

## Make Predictions on New Data

```python
import numpy as np

# Assume you have preprocessed EEG data: X_new (n_trials, channels, timepoints)
# IMPORTANT: Data must be in the SAME FORMAT as training data:
#   - Baseline corrected
#   - Z-score normalized
#   - Shape: (trials, timepoints, channels)
#   - Note: Data was transposed for CNN!

# Make predictions
predictions = model.predict(X_new)  # Output shape: (n_trials, 2)

# Get predicted class (argmax)
predicted_classes = np.argmax(predictions, axis=1)
predicted_probabilities = predictions  # [P(left_hand), P(right_hand)]

# Decode back to class names
class_names = metadata['class_names']
predicted_labels = [class_names[pred] for pred in predicted_classes]

# Get confidence for predicted class
confidence = np.max(predictions, axis=1)

for i in range(len(predicted_labels)):
    print(f"Trial {i}: {predicted_labels[i]} (confidence: {confidence[i]:.4f})")
```

## Model Details

**Architecture:**
- Input: 22 channels × 751 timepoints (motor imagery window)
- Temporal Conv: 40 filters, 100ms kernel
- Spatial Conv: 40 filters, 1x1 kernel
- Average Pooling: Reduce to 10 timepoints
- Dropout: 0.5
- Output: 2 classes (left_hand, right_hand)

**Performance:**
- Test Accuracy: 80.00%
- Test AUC-ROC: 0.9077
- Model Size: ~8 KB (.keras format)

**Key Requirements:**
1. Input data must be preprocessed identically to training data
2. Baseline correction: subtract first 100ms mean
3. Z-score normalization: per trial, per channel
4. Frequency range: Bandpass 8-30 Hz
5. Sampling rate: 250 Hz

**For Real-Time BCI Integration (Personne 2):**
```python
# Sliding window prediction for continuous decoding
window_size = 250  # samples (1 second at 250 Hz)
stride = 50        # samples (200ms overlap)

# This allows decoding motor intention at 5Hz frequency
for start_idx in range(0, data_length - window_size, stride):
    window = data[start_idx:start_idx + window_size]
    # Preprocess and predict...
    intention = model.predict(window)
```

**For Model Interpretability (Personne 3):**
- Temporal filters capture motor planning evolution (0.5-3.5s)
- Spatial filters learn contralateral motor cortex organization (C3/C4)
- High activations over mu band (8-13 Hz) → motor planning
- Use GradCAM for saliency maps of important channels

