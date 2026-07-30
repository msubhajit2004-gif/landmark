"""
src/inference.py
================

Purpose
-------
High-level inference wrapper that bundles model + encoder + config
into a single reusable :class:`LandmarkPredictor` class.

Why a class?
------------
* **Single responsibility** – the class owns the lifecycle of the
  loaded model and encoder, so callers don't have to manage three
  objects.
* **Reusable** – load once, predict many times (ideal for a web
  service or batch job).
* **Batch support** – :meth:`predict_batch` processes a list of paths
  efficiently.
* **Defensive** – handles missing models, corrupt images, and
  gracefully degrades.

Usage
-----
::

    from src.inference import LandmarkPredictor
    predictor = LandmarkPredictor()
    lm_id, conf, top5 = predictor.predict("photo.jpg")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from src.config import Config, get_config
from src.preprocessing import load_image, normalize_image
from src.utils import get_logger, load_encoder, load_model

logger = get_logger("inference")


class LandmarkPredictor:
    """Loaded-model wrapper for landmark prediction.

    Parameters
    ----------
    model_path : str, optional
        Path to the ``.keras`` model file.  Defaults to the config path.
    encoder_path : str, optional
        Path to the label encoder.  Defaults to the config path.
    config : Config, optional
        Injected config (for testing).  Loads from YAML if None.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        encoder_path: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.config = config or get_config()
        mp = Path(model_path) if model_path else Path(self.config.paths.models) / "landmark_vgg19_final.keras"
        ep = Path(encoder_path) if encoder_path else Path(self.config.paths.encoder_file)

        if not mp.exists():
            raise FileNotFoundError(
                f"Model not found: {mp}. Train the model first with `python -m src.train`."
            )
        if not ep.exists():
            raise FileNotFoundError(
                f"Label encoder not found: {ep}. Run training to generate it."
            )

        logger.info("Loading model from %s", mp)
        self.model = load_model(mp)
        logger.info("Loading encoder from %s", ep)
        self.encoder = load_encoder(ep)
        self.num_classes = len(self.encoder.classes_)
        logger.info("Predictor ready (%d classes)", self.num_classes)

    # ------------------------------------------------------------------
    # Single image
    # ------------------------------------------------------------------
    def predict(
        self,
        image_path: Union[str, Path],
    ) -> Tuple[int, float, List[Tuple[int, float]]]:
        """Predict the landmark in *image_path*.

        Returns
        -------
        (landmark_id, confidence, top5)
        """
        img = load_image(str(image_path), self.config.image.target_size)
        if img is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}. File missing or corrupt."
            )
        img = normalize_image(
            img,
            mode=self.config.image.normalize_mode,
            mean=tuple(self.config.image.mean),
        )
        batch = np.expand_dims(img, axis=0)
        probs = self.model.predict(batch, verbose=0)[0]
        return self._decode(probs)

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
    ) -> List[Tuple[int, float, List[Tuple[int, float]]]]:
        """Predict landmarks for a list of image paths.

        Corrupt images are skipped with a warning and return a sentinel
        ``(−1, 0.0, [])`` so the output list stays aligned with the input.
        """
        results = []
        valid_imgs = []
        valid_idx = []
        for i, p in enumerate(image_paths):
            img = load_image(str(p), self.config.image.target_size)
            if img is None:
                logger.warning("Skipping unreadable image: %s", p)
                results.append((-1, 0.0, []))
                continue
            img = normalize_image(
                img,
                mode=self.config.image.normalize_mode,
                mean=tuple(self.config.image.mean),
            )
            valid_imgs.append(img)
            valid_idx.append(i)

        if valid_imgs:
            batch = np.stack(valid_imgs, axis=0)
            probs_all = self.model.predict(batch, verbose=0)
            for j, probs in enumerate(probs_all):
                results[valid_idx[j]] = self._decode(probs)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _decode(self, probs: np.ndarray) -> Tuple[int, float, List[Tuple[int, float]]]:
        top5_idx = np.argsort(probs)[-5:][::-1]
        top5_probs = probs[top5_idx]
        top5_lm = self.encoder.inverse_transform(top5_idx)
        predicted_id = int(top5_lm[0])
        confidence = float(top5_probs[0])
        top5 = list(zip([int(x) for x in top5_lm], [float(p) for p in top5_probs]))
        return predicted_id, confidence, top5

    def label_to_landmark(self, label: int) -> int:
        """Decode a single integer class index to original landmark id."""
        return int(self.encoder.inverse_transform([label])[0])

    def landmark_to_label(self, landmark_id: int) -> int:
        """Encode an original landmark id to the model's class index."""
        return int(self.encoder.transform([landmark_id])[0])
