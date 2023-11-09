from typing import List, Tuple

from cv2.typing import MatLike
import torch

from ultralytics import YOLO


class YoloV8:
    def __init__(
        self,
        weight: str = None,
        conf: float = 0.25,
        iou: float = 0.3,
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
        result = self.model.predict(image, **self.config, classes=0, verbose=False)[0]

        outputs = []
        if result.boxes:
            for box in result.boxes:
                x1, y1, x2, y2 = (int(i.item()) for i in box.data[0][:4])

                outputs.append((x1, y1, x2, y2))

        return outputs
