from typing import List, Tuple
from functools import partial

from cv2.typing import MatLike
import torch

from ultralytics import YOLO


class YoloV8:
    def __init__(
        self,
        weight: str = None,
        conf: float = 0.25,
        iou: float = 0.3,
        track: bool = False,
        device: str = "auto",
    ):
        """
        Initialize the Yolo-v8 model

        Args:
            weight (str, optional): Path to the YOLO model weights file. Defaults to None.
            conf (float, optional): Confidence threshold for object detection. Defaults to 0.25.
            iou (float, optional): Intersection over Union (IoU) threshold. Defaults to 0.3.
            device (str, optional): Device on which to run the YOLO model ('auto', 'cuda', or 'cpu'). Defaults to "auto".
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.track = track
        self.model = YOLO(weight if weight else "weights/yolov8x.pt").to(self.device)
        self.config = {"conf": conf, "iou": iou}

    def __call__(self, image: MatLike) -> List[Tuple]:
        """
        Forward pass of the model

        Args:
            image (MatLike): Input image

        Returns:
            List: A list containing box of humans detected
        """
        return self.forward(image)

    def forward(self, image: MatLike) -> List[Tuple]:
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
                x1, y1, x2, y2, idx = (int(i.item()) for i in box.data[0][:5])
                outputs.append(
                    {"box": (x1, y1, x2, y2), "id": idx if self.track else None}
                )

        return outputs
