"""
src/predict.py
==============

Purpose
-------
Single-image prediction pipeline.  Loads the trained model and label
encoder, preprocesses any input image automatically, predicts the
landmark, decodes the prediction back to the original landmark id,
shows the confidence score and top-5 predictions, and optionally
visualises the result.

Run::

    python -m src.predict --image path/to/photo.jpg
    python -m src.predict --image path/to/photo.jpg --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.config import Config, get_config
from src.preprocessing import load_image, normalize_image
from src.utils import get_logger, load_model, load_encoder

logger = get_logger("predict")


def predict_image(
    image_path: str,
    model,
    encoder,
    config: Config,
) -> Tuple[int, float, List[Tuple[int, float]]]:
    """Run the full prediction pipeline on a single image.

    Returns
    -------
    (predicted_landmark_id, confidence, top5)
        ``top5`` is a list of ``(landmark_id, probability)`` tuples.
    """
    # 1. Load + preprocess
    img = load_image(image_path, config.image.target_size)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}. File missing or corrupt."
        )
    img = normalize_image(img, mode=config.image.normalize_mode, mean=tuple(config.image.mean))
    batch = np.expand_dims(img, axis=0)  # (1, H, W, 3)

    # 2. Predict
    probs = model.predict(batch, verbose=0)[0]  # (num_classes,)

    # 3. Decode
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_probs = probs[top5_idx]
    top5_lm = encoder.inverse_transform(top5_idx)

    predicted_id = int(top5_lm[0])
    confidence = float(top5_probs[0])
    top5 = list(zip([int(x) for x in top5_lm], [float(p) for p in top5_probs]))

    logger.info(
        "Predicted landmark %d with %.2f%% confidence",
        predicted_id, confidence * 100,
    )
    for i, (lm, p) in enumerate(top5):
        logger.info("  #%d: landmark %d (%.2f%%)", i + 1, lm, p * 100)

    return predicted_id, confidence, top5


def predict_and_visualize(
    image_path: str,
    config: Config,
    model=None,
    encoder=None,
    save_path: Optional[Path] = None,
) -> Path:
    """Predict and save a side-by-side image + top-5 bar chart."""
    from src.visualization import plot_top5_predictions

    if model is None:
        model = load_model(Path(config.paths.models) / "landmark_vgg19_final.keras")
    if encoder is None:
        encoder = load_encoder(config.paths.encoder_file)

    img = load_image(image_path, config.image.target_size)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_norm = normalize_image(img, mode=config.image.normalize_mode, mean=tuple(config.image.mean))
    batch = np.expand_dims(img_norm, axis=0)
    probs = model.predict(batch, verbose=0)[0]

    if save_path is None:
        save_path = Path(config.paths.figures) / f"prediction_{Path(image_path).stem}.png"
    return plot_top5_predictions(img_norm, probs, encoder, save_path=save_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict landmark from image")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--show", action="store_true", help="Save visualisation")
    args = parser.parse_args()

    config = get_config(Path(args.config) if args.config else None)
    model_path = args.model or Path(config.paths.models) / "landmark_vgg19_final.keras"
    model = load_model(model_path)
    encoder = load_encoder(config.paths.encoder_file)

    predicted_id, conf, top5 = predict_image(args.image, model, encoder, config)

    print(f"\nPredicted landmark: {predicted_id}")
    print(f"Confidence: {conf*100:.2f}%")
    print("Top-5 predictions:")
    for i, (lm, p) in enumerate(top5):
        print(f"  {i+1}. Landmark {lm} — {p*100:.2f}%")

    if args.show:
        path = predict_and_visualize(args.image, config, model, encoder)
        print(f"Visualisation saved to: {path}")


if __name__ == "__main__":
    main()
