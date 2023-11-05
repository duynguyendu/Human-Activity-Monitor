from typing import Dict, Tuple

from cv2.typing import MatLike
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
        margin: int = 0,
        add_border: bool = True,
        device: str = "auto",
    ):
        """
        Initialize the Yolo-v8 model

        Args:
            weight (str, optional): Path to the YOLO model weights file. Defaults to None.
            conf (float, optional): Confidence threshold for object detection. Defaults to 0.25.
            iou (float, optional): Intersection over Union (IoU) threshold. Defaults to 0.3.
            size (Tuple, optional): Size to which the detected humans will be resized. Defaults to (224, 224).
            margin (int, optional): Margin to apply when cropping humans from the original image. Defaults to 0.
            add_border (bool, optional): Whether to add a border to the cropped human images. Defaults to True.
            device (str, optional): Device on which to run the YOLO model ('auto', 'cuda', or 'cpu'). Defaults to "auto".
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.size = tuple_handler(size, max_dim=2)
        self.margin = margin
        self.border = add_border
        self.model = YOLO(weight if weight else "weights/yolov8x.pt").to(self.device)
        self.config = {"conf": conf, "iou": iou}

    def __call__(self, X: MatLike) -> Dict:
        """
        Forward pass of the model

        Args:
            X (MatLike): Input tensor

        Returns:
            Dict: A dictionary containing information about detected humans
        """
        return self.forward(X)

    def forward(self, X: MatLike) -> Dict:
        """
        Forward pass of the model

        Args:
            X (MatLike): Input tensor

        Returns:
            Dict: A dictionary containing information about detected humans
        """
        result = self.model.predict(X, **self.config, classes=0, verbose=False)[0]

        outputs = []
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = (int(i.item()) for i in box.data[0][:4])

                # Apply margin
                weight = result.orig_img.shape[1]
                x1 = max(0, x1 - self.margin)
                y1 = max(0, y1 - self.margin)
                x2 = min(weight, x2 + self.margin)
                y2 = min(weight, y2 + self.margin)

                # Cut out human
                human = result.orig_img[y1:y2, x1:x2]

                if self.border:
                    human = ImageProcessing.add_border(human)

                human = cv2.resize(human, self.size)

                outputs.append({"box": (x1, y1, x2, y2), "human": human})

        return outputs
