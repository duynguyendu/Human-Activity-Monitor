from typing import Tuple

import torch
import cv2

from ultralytics import YOLO

from modules.data import ImageProcessing
from modules.utils import tuple_handler


class YoloV8:
    def __init__(
        self,
        weight: str = None,
        conf: float = 0.25,
        iou: float = 0.3,
        size: Tuple = (224, 224),
        add_border: bool = True,
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.size = tuple_handler(size, max_dim=2)
        self.border = add_border
        self.model = YOLO(weight if weight else "weights/yolov8x.pt").to(self.device)
        self.config = {"conf": conf, "iou": iou}

    def __call__(self, X: torch.Tensor) -> str:
        return self.predict(X)

    def predict(self, X: torch.Tensor) -> str:
        result = self.model.predict(X, **self.config, classes=0, verbose=False)[0]

        outputs = []
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = (int(i.item()) for i in box.data[0][:4])

                human = result.orig_img[y1:y2, x1:x2]

                if self.border:
                    human = ImageProcessing.add_border(human)

                human = cv2.resize(human, self.size)

                outputs.append({"box": (x1, y1, x2, y2), "human": human})

        return outputs
