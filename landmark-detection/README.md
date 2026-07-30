# Google Landmark Detection — VGG19 Transfer Learning

A production-ready, modular deep-learning system that recognises famous
landmarks from photographs using **VGG19** transfer learning with
TensorFlow / Keras.

The project is organised so that every concern — configuration,
preprocessing, augmentation, model definition, training, evaluation,
visualisation, and inference — lives in its own module.  All
hyper-parameters are driven by a single `config.yaml`, so you can
experiment without editing source code.

---

## Table of Contents
1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Prediction / Inference](#prediction--inference)
9. [Troubleshooting](#troubleshooting)
10. [Hardware Requirements](#hardware-requirements)
11. [Machine-Learning Concepts](#machine-learning-concepts)

---

## Features

- **Transfer learning** with pretrained VGG19 (ImageNet weights).
- **Two-phase training**: frozen-backbone classifier training followed
  by fine-tuning of the last convolutional block.
- **GPU + mixed-precision** support with automatic CPU fallback.
- **Efficient `tf.data` pipelines** — caching, shuffling, prefetching,
  parallel loading.
- **Advanced augmentation** via Keras preprocessing layers (GPU-native).
- **Comprehensive evaluation**: accuracy, top-5 accuracy, precision,
  recall, F1, confusion matrix, per-class report, error analysis.
- **Visualisation**: training curves, confusion matrix, prediction
  gallery, top-5 confidence bars.
- **Defensive programming**: corrupt-image skipping, missing-file
  handling, clear error messages.
- **Reproducible**: single-seed control, deterministic ops.
- **Reusable inference class** (`LandmarkPredictor`) for batch or
  single-image prediction.

---

## Project Structure

```
landmark-detection/
├── config.yaml              # All hyper-parameters
├── requirements.txt
├── README.md
├── data/
│   ├── raw/                 # Place train.csv + image folders here
│   ├── processed/           # Generated train/val/test CSV splits
│   ├── train/  val/  test/   # (optional) organised image dirs
├── notebooks/
│   └── landmark_eda.ipynb   # Exploratory analysis notebook
├── models/                  # Saved .keras models + label encoder
├── checkpoints/             # Per-phase best-weight checkpoints
├── outputs/
│   ├── figures/             # Confusion matrices, prediction galleries
│   └── plots/               # Training curves, EDA plots
├── logs/                    # Rotating log files + TensorBoard logs
└── src/
    ├── __init__.py
    ├── config.py            # YAML config loader (dataclasses)
    ├── utils.py             # Logging, seeds, GPU, save/load
    ├── preprocessing.py     # Image loading, resizing, normalisation
    ├── augmentation.py      # Keras preprocessing-layer augmentation
    ├── dataset.py           # CSV loading, EDA, encoding, splits, tf.data
    ├── model.py             # VGG19 backbone + custom head, compile, fine-tune
    ├── callbacks.py         # EarlyStopping, ReduceLR, Checkpoint, TensorBoard
    ├── metrics.py           # Custom metrics, per-class report, error stats
    ├── train.py             # Two-phase training orchestration
    ├── evaluate.py          # Full evaluation + report generation
    ├── visualization.py     # All plotting utilities
    ├── predict.py           # Single-image CLI prediction
    └── inference.py         # Reusable LandmarkPredictor class
```

---

## Dataset

This project targets the **Google Landmarks Dataset v2** (GLDv2):

- ~5 million images
- ~200,000 landmark classes
- Metadata CSV (`train.csv`) with columns `id` (image id) and
  `landmark_id` (integer class).
- Images stored in a 3-level folder hierarchy derived from the first
  three characters of the image id:
  ```
  data/raw/0a/1b/2c/0a1b2c3d....jpg
  ```

### Advantages
- Massive scale → rich feature learning.
- Real-world noise (blur, occlusion, viewpoint variation).

### Limitations
- **Extreme long-tail class imbalance** — a handful of landmarks have
  thousands of photos while most have fewer than ten.
- Some images are mislabelled or corrupt.
- Full dataset is ~500 GB — impractical on commodity hardware.

### Manageable subset
Because of the above, this project keeps only the **top-K most frequent
classes** (default 50) and caps each class at `max_samples_per_class`
(default 500) images.  This produces a balanced, trainable subset while
preserving the real-world challenges.  Adjust `dataset.top_k_classes`
and `dataset.max_samples_per_class` in `config.yaml`.

### Obtaining the data
1. Download `train.csv` from the [GLDv2 Kaggle page](https://www.kaggle.com/google/google-landmark-dataset).
2. Download the image archives (or a subset).
3. Place `train.csv` at `data/raw/train.csv`.
4. Extract images under `data/raw/` preserving the 3-level hierarchy.

---

## Installation

```bash
# 1. Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) verify TensorFlow sees your GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Configuration

All settings live in `config.yaml`.  Key sections:

| Section      | Key parameters |
|--------------|----------------|
| `image`      | `height`, `width`, `normalize_mode` (`tf` / `caffe` / `scale`) |
| `dataset`    | `top_k_classes`, `min/max_samples_per_class`, `val/test_split` |
| `training`   | `batch_size`, `epochs_phase1/2`, `phase1/2_lr`, `optimizer`, `mixed_precision` |
| `model`      | `backbone`, `weights`, `dropout`, `dense_units`, `label_smoothing` |
| `augmentation`| `random_rotation`, `random_zoom`, `random_brightness`, etc. |

---

## Training

```bash
# Full two-phase training (frozen + fine-tune)
python -m src.train

# Override epochs, skip fine-tuning
python -m src.train --epochs1 10 --epochs2 5
python -m src.train --no-finetune
```

- **Phase 1** trains only the custom classifier head (backbone frozen)
  with Adam at lr=1e-3.
- **Phase 2** unfreezes the last 4 VGG19 layers and fine-tunes at
  lr=1e-4 to adapt high-level features to landmarks.
- Best weights (by validation accuracy) are saved to `checkpoints/`.
- The final model is saved to `models/landmark_vgg19_final.keras`.
- Training curves are plotted to `outputs/plots/training_history.png`.

---

## Evaluation

```bash
python -m src.evaluate --model models/landmark_vgg19_final.keras
```

Produces:
- Accuracy, top-5 accuracy, macro/weighted F1.
- Per-class precision / recall / F1 / support.
- Confusion matrix heatmap (`outputs/figures/confusion_matrix_test.png`).
- Error statistics (most confused pairs, overconfidence).
- Prediction gallery (`outputs/figures/predictions_test.png`).
- JSON report (`outputs/figures/report_test.json`).

---

## Prediction / Inference

### Single image (CLI)
```bash
python -m src.predict --image path/to/photo.jpg --show
```
Prints the predicted landmark id, confidence, and top-5 predictions.
With `--show` it saves a side-by-side image + bar chart.

### Programmatic (batch)
```python
from src.inference import LandmarkPredictor

predictor = LandmarkPredictor()
lm_id, confidence, top5 = predictor.predict("photo.jpg")
results = predictor.predict_batch(["a.jpg", "b.jpg", "c.jpg"])
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Model not found` | Run `python -m src.train` first. |
| `Label encoder not found` | Training generates it; ensure training completed. |
| OOM on GPU | Reduce `training.batch_size` in config; enable `mixed_precision`. |
| Very slow first epoch | Expected — `tf.data` caches after first pass. |
| `Corrupt image` warnings | Normal — the pipeline skips unreadable files automatically. |
| CUDA / driver mismatch | Install the CUDA version matching your TensorFlow release. |
| `tf-keras` import error | `pip install tf-keras==2.15.0` (compatibility shim). |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|--------|-------------|
| CPU       | 4 cores | 8+ cores |
| RAM       | 8 GB   | 16-32 GB |
| GPU       | Optional | NVIDIA with 8 GB VRAM (e.g. RTX 3060) |
| Disk      | 20 GB  | 100+ GB for full dataset subset |

**GPU recommendations**: Any CUDA-enabled NVIDIA GPU dramatically
speeds training.  Mixed precision (FP16) gives ~2× speed-up on
Tensor-Core GPUs (RTX 20-series and newer).  The code automatically
detects GPUs and falls back to CPU if none is present.

---

## Machine-Learning Concepts

- **Supervised learning** — we train on labelled (image, landmark_id)
  pairs to learn a mapping from pixels to class.
- **CNN** — convolutional neural networks use learnable filters to
  detect spatial patterns (edges → textures → object parts).
- **Convolution** — a sliding dot-product that produces feature maps.
- **Pooling** — downsampling that adds translation invariance.
- **Transfer learning** — reusing features learned on ImageNet so we
  need far less data and compute.
- **GlobalAveragePooling2D** — collapses spatial dims to one value per
  channel, reducing parameters vs Flatten.
- **Softmax** — converts logits to a probability distribution over
  classes.
- **Cross-entropy loss** — measures the divergence between predicted
  probabilities and the true label.
- **Adam optimizer** — adaptive moment estimation; combines momentum
  and per-parameter learning rates.
- **SGD with momentum** — classical optimizer; sometimes more stable
  for fine-tuning.
- **Dropout** — randomly zeroes activations during training to
  prevent co-adaptation (regularisation).
- **BatchNormalization** — normalises activations to stabilise and
  speed up training.
- **Overfitting / underfitting** — model memorises vs fails to learn;
  augmentation, dropout and early stopping fight over-fitting.
- **Class imbalance** — some classes have far more samples; we cap
  per-class counts and use stratified splits.
- **Data augmentation** — synthetic variations expand the data
  distribution and teach invariance.
- **Top-5 accuracy** — fraction of samples whose true label is in the
  model's top-5 predictions; useful when many landmarks look alike.
- **Fine-tuning** — unfreezing upper layers and retraining at a small
  learning rate to adapt pre-trained features to the target domain.
