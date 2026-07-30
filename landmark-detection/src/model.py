"""
src/model.py
============

Purpose
-------
Build and compile the landmark-classification model based on **VGG19**
transfer learning.

What is VGG19?
--------------
VGG19 is a 19-layer (16 conv + 3 dense, plus 5 pooling) convolutional
neural network designed by the Visual Geometry Group at Oxford.  Its
architecture is deliberately simple and uniform:

* **5 convolutional blocks** (conv1 → conv5), each containing 2-4
  stacked 3×3 convolutions followed by a 2×2 max-pooling layer.
* Block depths double the channel count: 64 → 128 → 256 → 512 → 512.
* The original head ends with two 4096-unit dense layers and a 1000-way
  softmax trained on ImageNet.

Why transfer learning?
----------------------
Training a 19-layer CNN from scratch on millions of images requires
enormous data and compute.  Because VGG19 was pre-trained on ImageNet
(1.2M images, 1000 classes), its early/middle layers already encode
generic visual features – edges, textures, shapes, object parts – that
transfer to almost any vision task.  We reuse these features and only
train a new classifier head on top, which converges in minutes instead
of days and needs far less data.

Architecture of our custom head
--------------------------------
1. **VGG19 backbone** (``include_top=False``) → outputs a
   (7, 7, 512) feature map.
2. **GlobalAveragePooling2D** – collapses each 512-channel feature map
   to a single value, giving a 512-d vector.  Compared to ``Flatten``
   (25k-d), GAP drastically reduces parameters and over-fitting.
3. **Dense(1024) + BatchNorm + ReLU + Dropout(0.5)** – a regularised
   fully-connected block.
4. **Dense(num_classes, softmax)** – probability distribution over
   landmark classes.

Two-phase training
------------------
* **Phase 1 – frozen backbone**: only the custom head trains (fast,
  large LR).
* **Phase 2 – fine-tuning**: we unfreeze the last few VGG19 layers and
  retrain with a 10× smaller LR so the pre-trained features adapt to
  landmarks without being destroyed.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

from src.config import Config
from src.utils import get_logger

logger = get_logger("model")


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
def build_backbone(config: Config) -> tf.keras.Model:
    """Load the VGG19 backbone with ImageNet weights, no top.

    Raises a clear error if the backbone name is unsupported.
    """
    backbone_name = config.model.backbone.lower()
    kwargs = dict(
        include_top=config.model.include_top,
        weights=config.model.weights,
        input_shape=(config.image.height, config.image.width, config.image.channels),
        pooling=None,
    )

    if backbone_name == "vgg19":
        from tensorflow.keras.applications import VGG19
        backbone = VGG19(**kwargs)
    elif backbone_name == "vgg16":
        from tensorflow.keras.applications import VGG16
        backbone = VGG16(**kwargs)
    elif backbone_name == "resnet50":
        from tensorflow.keras.applications import ResNet50
        backbone = ResNet50(**kwargs)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_name}")

    logger.info("Loaded %s backbone (%s weights)", backbone_name, config.model.weights)
    return backbone


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
def build_model(config: Config, num_classes: int) -> tf.keras.Model:
    """Assemble the full transfer-learning model.

    Returns an uncompiled model; call :func:`compile_model` separately
    so the same architecture can be compiled with different optimisers
    (e.g. phase-1 Adam vs phase-2 SGD).
    """
    backbone = build_backbone(config)
    backbone.trainable = False  # freeze for phase 1

    inputs = tf.keras.Input(
        shape=(config.image.height, config.image.width, config.image.channels),
        name="input_image",
    )

    # Preprocessing inside the graph keeps inference consistent.
    # VGG19 expects images in [0, 255] with ImageNet mean subtracted
    # (BGR order).  We expose a toggle via config.image.normalize_mode.
    if config.image.normalize_mode == "caffe":
        mean = tf.constant(config.image.mean, dtype=tf.float32)  # BGR
        x = layers.Lambda(
            lambda z: tf.gather(z, [2, 1, 0], axis=-1) - mean,  # RGB→BGR
            name="caffe_preprocess",
        )(inputs)
    else:
        x = layers.Lambda(lambda z: z / 255.0, name="tf_preprocess")(inputs)

    x = backbone(x, training=False)

    # Head
    if config.model.use_global_avg_pool:
        x = layers.GlobalAveragePooling2D(name="gap")(x)
    else:
        x = layers.Flatten(name="flatten")(x)

    x = layers.Dense(config.model.dense_units, name="fc1")(x)
    if config.model.use_batchnorm:
        x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.Dropout(config.model.dropout, name="dropout1")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs, outputs, name="landmark_vgg19")
    logger.info(
        "Built model: %d params (backbone frozen=%s)",
        model.count_params(),
        not backbone.trainable,
    )
    return model


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------
def get_optimizer(config: Config, lr: float):
    """Return the configured optimizer (Adam or SGD-with-momentum)."""
    name = config.training.optimizer.lower()
    if name == "adam":
        return optimizers.Adam(learning_rate=lr)
    if name == "sgd":
        return optimizers.SGD(
            learning_rate=lr,
            momentum=config.training.sgd_momentum,
            nesterov=config.training.sgd_nesterov,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def compile_model(
    model: tf.keras.Model,
    config: Config,
    lr: Optional[float] = None,
    num_classes: Optional[int] = None,
) -> tf.keras.Model:
    """Compile *model* with loss, optimizer and metrics.

    Uses ``SparseCategoricalCrossentropy`` because our labels are
    integer-encoded (not one-hot).  Label smoothing (0.1) prevents the
    model from becoming over-confident and improves calibration.

    Tracked metrics: accuracy, top-5 accuracy, precision, recall, AUC.
    """
    if lr is None:
        lr = config.training.phase1_lr
    optimizer = get_optimizer(config, lr)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        label_smoothing=config.model.label_smoothing,
    )

    metrics = [
        "accuracy",
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=config.evaluation.top_k, name="top5_accuracy",
        ),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]
    # AUC requires one-hot labels in older TF; use sparse-safe variant.
    try:
        metrics.append(tf.keras.metrics.AUC(name="auc", multi_label=False))
    except Exception:  # pragma: no cover
        pass

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    logger.info("Compiled model (lr=%g, optimizer=%s)", lr, config.training.optimizer)
    return model


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def unfreeze_backbone(
    model: tf.keras.Model,
    config: Config,
    num_layers: Optional[int] = None,
) -> tf.keras.Model:
    """Unfreeze the last *num_layers* layers of the VGG19 backbone for
    fine-tuning, then recompile with a smaller learning rate.

    Fine-tuning rationale: the last conv block learns higher-level
    features (object parts).  Adapting them to landmarks – while keeping
    the earlier generic layers frozen – often yields a few extra
    accuracy points.  We use a 10× smaller LR to avoid destroying the
    pre-trained weights.
    """
    if num_layers is None:
        num_layers = config.training.fine_tune_layers

    # The backbone is the first layer after the input/preprocessing.
    backbone = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            backbone = layer
            break
    if backbone is None:
        logger.warning("No backbone sub-model found; cannot unfreeze.")
        return model

    backbone.trainable = True
    # Freeze all but the last N layers.
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False

    # Recompile with phase-2 LR.
    compile_model(model, config, lr=config.training.phase2_lr)
    logger.info(
        "Unfroze last %d backbone layers for fine-tuning (lr=%g)",
        num_layers, config.training.phase2_lr,
    )
    return model
