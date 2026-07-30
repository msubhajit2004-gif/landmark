"""
src/augmentation.py
===================

Purpose
-------
Advanced, GPU-accelerated data augmentation for the landmark-detection
training pipeline.  We use Keras **preprocessing layers** (not the
legacy ``ImageDataGenerator``) because:

1. They run on the GPU as part of the model graph → faster training.
2. They are serialisable with the model → no separate augmentation
   pipeline at inference time (we can simply omit the layers).
3. They integrate natively with ``tf.data``.

Why augmentation improves generalisation
-----------------------------------------
A CNN learns invariants from data.  If every training image of a
landmark is perfectly framed, the network never sees the landmark
under different lighting, viewpoints, or crops, and will over-fit to
those exact conditions.  By synthetically expanding the data
distribution – flips, rotations, brightness shifts, zooms – we teach
the network that "Eiffel Tower at dusk" and "Eiffel Tower tilted 15°"
are the same class.  This is a form of *regularisation* and is
especially valuable for the long-tailed, class-imbalanced Google
Landmarks dataset.

Layers used
-----------
* ``RandomFlip``          – horizontal flip (landmarks are rarely
  handedness-specific).
* ``RandomRotation``      – ±20° rotation.
* ``RandomZoom``          – random zoom-in/zoom-out.
* ``RandomTranslation``   – width & height shift.
* ``RandomBrightness``    – simulate varying exposure.
* ``RandomContrast``      – simulate varying lighting conditions.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf

from src.config import Config
from src.utils import get_logger

logger = get_logger("augmentation")


def build_augmentation_layers(config: Config) -> tf.keras.Sequential:
    """Return a ``Sequential`` model of Keras preprocessing layers.

    The returned object can be:
    * inserted directly into a ``tf.keras.Sequential`` model, or
    * applied inside a ``tf.data`` map function.

    All layers operate in the [0, 1] or [0, 255] float32 space – they
    are placed *before* normalisation in the pipeline.
    """
    aug = config.augmentation
    layers = []

    if not aug.enabled:
        logger.info("Augmentation disabled in config")
        return tf.keras.Sequential([], name="augmentation_disabled")

    # Horizontal flip – safe for most landmarks.
    layers.append(tf.keras.layers.RandomFlip(aug.random_flip))

    # Rotation in degrees.
    if aug.random_rotation > 0:
        layers.append(tf.keras.layers.RandomRotation(
            factor=aug.random_rotation / 360.0,
            fill_mode="reflect",
        ))

    # Zoom.
    if aug.random_zoom > 0:
        layers.append(tf.keras.layers.RandomZoom(
            height_factor=(-aug.random_zoom, aug.random_zoom),
            width_factor=(-aug.random_zoom, aug.random_zoom),
            fill_mode="reflect",
        ))

    # Width / height translation.
    if aug.random_width_shift > 0 or aug.random_height_shift > 0:
        layers.append(tf.keras.layers.RandomTranslation(
            height_factor=aug.random_height_shift,
            width_factor=aug.random_width_shift,
            fill_mode="reflect",
        ))

    # Brightness.
    if aug.random_brightness > 0:
        layers.append(tf.keras.layers.RandomBrightness(
            factor=aug.random_brightness,
        ))

    # Contrast.
    if aug.random_contrast > 0:
        layers.append(tf.keras.layers.RandomContrast(
            factor=aug.random_contrast,
        ))

    seq = tf.keras.Sequential(layers, name="augmentation")
    logger.info("Built augmentation pipeline with %d layers", len(layers))
    return seq


def augment_image(
    image: tf.Tensor,
    augmentation_model: tf.keras.Sequential,
) -> tf.Tensor:
    """Apply augmentation to a single image tensor.

    Intended for use inside ``tf.data.Dataset.map``.  The image is
    expected to be float32 in [0, 1] (i.e. after ``"tf"`` normalisation)
    or [0, 255].
    """
    # Add batch dimension, apply, remove batch dimension.
    image = tf.expand_dims(image, 0)
    image = augmentation_model(image, training=True)
    return tf.squeeze(image, axis=0)


# ---------------------------------------------------------------------------
# Legacy ImageDataGenerator support (optional)
# ---------------------------------------------------------------------------
def build_datagen(config: Config):
    """Return a Keras ``ImageDataGenerator`` for users who prefer the
    legacy API (e.g. when mixing with flow_from_directory).

    This is provided for completeness; the main pipeline uses
    :func:`build_augmentation_layers`.
    """
    aug = config.augmentation
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    return ImageDataGenerator(
        rotation_range=aug.random_rotation,
        width_shift_range=aug.random_width_shift,
        height_shift_range=aug.random_height_shift,
        zoom_range=aug.random_zoom,
        horizontal_flip=aug.random_flip == "horizontal",
        brightness_range=(1 - aug.random_brightness, 1 + aug.random_brightness),
        fill_mode="reflect",
    )
