from typing import Union
from pathlib import Path
import os

from boxmot import OCSORT, DeepOCSORT
import numpy as np

from .utils import device_handler


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

        # Check if the provided weight path exists
        if weight and not os.path.exists(weight):
            raise FileNotFoundError(weight)

        # Setup model
        self.model = self._setup_model(
            weight, fp16, det_conf, det_iou, max_age, min_hits, device_handler(device)
        )

    def _setup_model(
        self,
        weight: str,
        fp16: bool,
        det_conf: float,
        det_iou: float,
        max_age: int,
        min_hits: int,
        device: str,
    ) -> Union[OCSORT, DeepOCSORT]:
        """
        Set up the tracking model.

        Args:
            weight (str): Path to the pre-trained model weight file.
            fp16 (bool): Whether to use FP16 precision.
            det_conf (float): Detection confidence threshold.
            det_iou (float): Detection IOU (Intersection over Union) threshold.
            max_age (int): Maximum number of frames to keep tracking after losing detection.
            min_hits (int): Minimum number of detection hits to initiate tracking.
            device (str): Device to run the tracker on.

        Returns:
            Union[OCSORT, DeepOCSORT]: An instance of the tracking model.
        """

        # Save config
        config = {
            "max_age": max_age,
            "min_hits": min_hits,
            "det_thresh": det_conf,
            "iou_threshold": det_iou,
            "asso_func": "ciou",
        }

        # Initialize model based on weight provided
        if weight:
            tracker = DeepOCSORT(
                model_weights=Path(weight), device=device, fp16=fp16, **config
            )
        else:
            tracker = OCSORT(**config)

        return tracker

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
