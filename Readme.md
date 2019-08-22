# ID R&D Voice Antispoofing Challenge

2nd place solution for [ID R&D Voice Antispoofing Challenge](https://boosters.pro/championship/idrnd_antispoof/overview).

### Task

Binary classification of sound recordings. The task was to distinguish the human voice from the spoofed speech.

### Description

All data files (human and spoof) are expected to be in the same folder data/train.

Main files:
- `src/prepare_metadata.py` - prepare csv with metadata
- `src/train.py` - train all models
- `src/predict.py` - prediction on test set (not reproducible, should run on competition platform)

Trained models go to models/, log file goes to logs/.

## Approach

- Blending (simple average) of 5 different models with different preprocessing.
- Mel-spectrogram and constant-Q transform with different parameters for preprocessing.
- Small and simple 2D and 1D CNN models - several convolutional blocks, global max pooling and dense layer in the end, batch normalization.
- There was 100 mb solution size limit, so using custom lightweight models helps to blend more models (and allows to make cross-validation and blend trained models from several folds).
- Prepare all spectrograms in advance (not on-the-fly), then train models, this is especially useful for constant-Q transform, which is quite slow.

All parameters of trained models and data preprocessing are in `src/config.yaml`.
