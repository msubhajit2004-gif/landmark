"""
src package – Google Landmark Detection.

Public API:
    from src.config import get_config
    from src.dataset import prepare_dataset
    from src.model import build_model, compile_model, unfreeze_backbone
    from src.train import main as train_main
    from src.evaluate import main as evaluate_main
    from src.predict import predict_image
    from src.inference import LandmarkPredictor
"""

__version__ = "1.0.0"
