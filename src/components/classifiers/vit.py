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

    def __check_dim(self, X: torch.Tensor) -> torch.Tensor:
        match X.dim():
            case 3:
                X = X.unsqueeze(0)
            case 4:
                pass
            case _:
                raise ValueError(f"Input dimension must be 4. Got {X.dim()} instead.")
        return X

    def predict(self, X: torch.Tensor) -> str:
        with torch.inference_mode():
            X = self.__check_dim(X).to(self.device)
            outputs = self.model(X)
            _, pred = torch.max(outputs, 1)

        return self.CLASESS[pred.item()]
