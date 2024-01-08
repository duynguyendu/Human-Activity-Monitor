from pathlib import Path
import os

from boxmot import OCSORT, DeepOCSORT
import numpy as np

from .utils import device_handler, check_half


class Tracker:
    def __init__(
        self,
        weight: str = None,
        fp16: bool = True,
        det_conf: float = 0.3,
        det_iou: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        device: str = "auto",
    ) -> None:
        """
        Initialize a Tracker object.

        Args:
            weight (str, optional): Path to the pre-trained model weight file. Defaults to None.
            fp16 (bool, optional): Whether to use FP16 precision. Defaults to True.
            det_conf (float, optional): Detection confidence threshold. Defaults to 0.3.
            det_iou (float, optional): Detection IOU (Intersection over Union) threshold. Defaults to 0.3.
            max_age (int, optional): Maximum number of frames to keep tracking after losing detection. Defaults to 30.
            min_hits (int, optional): Minimum number of detection hits to initiate tracking. Defaults to 3.
            device (str, optional): Device to run the tracker on, "auto" for automatic selection. Defaults to "auto".

        Raises:
            FileNotFoundError: If the specified weight file is not found.
        """

        # Save config
        self.config = {
            "max_age": max_age,
            "min_hits": min_hits,
            "det_thresh": det_conf,
            "iou_threshold": det_iou,
            "asso_func": "ciou",
        }

        # Deep Learning model
        if weight:
            if not os.path.exists(weight):
                raise FileNotFoundError(weight)

            self.device = device_handler(device)

            self.model = DeepOCSORT(
                Path(weight), device, check_half(fp16, self.device), **self.config
            )

        # Algorithm based model
        else:
            self.model = OCSORT(**self.config)

    def update(self, dets: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Update the tracking model with new detections in the given frame.

        Args:
            dets (np.ndarray): Array containing detection information.
            image (np.ndarray): The input image frame.

        Returns:
            np.ndarray: Updated tracking results.
        """
        return self.model.update(dets, image)
