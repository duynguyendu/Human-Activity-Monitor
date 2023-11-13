from typing import List, Tuple, Union
from functools import partial

from rich import print
import numpy as np
import torch
import cv2

from ultralytics import YOLO


class YoloV8:
    def __init__(
        self,
        weight: str = None,
        conf: float = 0.25,
        iou: float = 0.7,
        size: int | Tuple = 640,
        half: bool = False,
        track: bool = False,
        device: str = "auto",
    ):
        """
        Initialize the Yolo-v8 model

        Args:
            weight (str, optional): Path to the YOLO model weights file. Defaults to None.
            conf (float, optional): Confidence threshold for object detection. Defaults to 0.25.
            iou (float, optional): Intersection over Union (IoU) threshold. Defaults to 0.3.
            size (int or Tuple, optional): Input size for the YOLO model. Defaults to 640.
            half (bool, optional): Use half precision (float16) for inference. Defaults to False.
            track (bool, optional): Enable object tracking. Defaults to False.
            device (str, optional): Device to run the model ('auto', 'cuda', or 'cpu'). Defaults to "auto".
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.track = track
        self.model = YOLO(weight if weight else "weights/yolov8x.pt").to(self.device)
        if half and device == "cpu":
            print(
                "[yellow][WARNING] Yolo-v8: Half is only supported on CUDA. Using default float32.[/]"
            )
            half = False
        self.config = {"conf": conf, "iou": iou, "imgsz": size, "half": half}

    def __call__(self, image: Union[cv2.Mat, np.ndarray]) -> List[Tuple]:
        """
        Forward pass of the model

        Args:
            image (MatLike): Input image

        Returns:
            List: A list containing box of humans detected
        """
        return self.forward(image)

    def forward(self, image: Union[cv2.Mat, np.ndarray]) -> List[Tuple]:
        """
        Forward pass of the model

        Args:
            image (MatLike): Input image

        Returns:
            List: A list containing box of humans detected
        """
        model = self.model.track if self.track else self.model.predict

        model = partial(model, source=image, classes=0, **self.config, verbose=False)

        result = (
            model(
                persist=True,
                tracker="configs/tracker.yaml",
            )
            if self.track
            else model()
        )[0]

        outputs = []
        if result.boxes:
            for box in result.boxes:
                data = [i.item() for i in box.data[0][:6]]

                x1, y1, x2, y2 = (int(i) for i in data[:4])

                human = {"box": (x1, y1, x2, y2)}

                A, B = data[4:]

                human.update({"id": int(A), "conf": B} if self.track else {"conf": A})

                outputs.append(human)

        return outputs
