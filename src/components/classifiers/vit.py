import torch
import os

from modules.model import LitModel
from models import ViT as vit


class ViT:
    CLASESS = ["idle", "laptop", "phone", "walk"]

    def __init__(self, checkpoint: str, device: str = "auto"):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        self.ckpt = checkpoint

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model = LitModel(
            model=vit("B_32", num_classes=len(self.CLASESS)), checkpoint=self.ckpt
        ).to(self.device)

    def __call__(self, X: torch.Tensor) -> str:
        return self.predict(X)

    def predict(self, X: torch.Tensor) -> str:
        with torch.inference_mode():
            outputs = self.model(X)
            _, pred = torch.max(outputs, 1)

        return self.CLASESS[pred.item()]
