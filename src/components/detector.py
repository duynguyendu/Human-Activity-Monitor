from typing import Dict, List, Tuple, Union
from functools import partial

from torch.nn import Module
from rich import print
import numpy as np
import torch
import cv2

from ultralytics import YOLO

from src.modules.utils import device_handler


class Detector:
    def __init__(
        self,
        weight: str = "weights/yolov8x.pt",
        track: bool = False,
        conf: float = 0.25,
        iou: float = 0.7,
        size: Union[int, Tuple] = 640,
        half: bool = False,
        fuse: bool = False,
        onnx: bool = False,
        optimize: bool = False,
        backend: str = None,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize the Yolo-v8 model

        Args:
            weight (str, optional): Path to the YOLO model weights file. Defaults to None.
            track (bool, optional): Enable object tracking. Defaults to False.
            conf (float, optional): Confidence threshold for object detection. Defaults to 0.25.
            iou (float, optional): Intersection over Union (IoU) threshold. Defaults to 0.3.
            size (int or Tuple, optional): Input size for the YOLO model. Defaults to 640.
            half (bool, optional): Use half precision (float16) for inference. Defaults to False.
            fuse (bool, optional): Fuse model layer. Defaults to False.
            onnx (bool, optional): Using onnx model. Defaults to False.
            optimize (bool, optional): Use TorchDynamo for model optimization. Defaults to False.
            backend (str, optional): Backend to be used for model optimization. Defaults to None.
            device (str, optional): Device to run the model ('auto', 'cuda', or 'cpu'). Defaults to "auto".
        """
        self.device = device_handler(device)
        self.model = self.__get_model(
            weight=weight,
            track=track,
            fuse=fuse,
            format="onnx" if onnx else "pt",
            optimize=optimize,
            backend=backend,
            config={
                "conf": conf,
                "iou": iou,
                "imgsz": size,
                "half": self.__check_half(half),
                "device": self.device,
            },
            **kwargs
        )

    def __call__(self, image: Union[cv2.Mat, np.ndarray]) -> List[Tuple]:
        """
        Forward pass of the model

        Args:
            image (MatLike): Input image

        Returns:
            List: A list containing box of humans detected
        """
        return self.forward(image)

    def __onnx_model(self):
        ...

    def __tensorrt_model(self):
        ...

    def __compile(self, X: Module, backend: str) -> Module:
        """
        Compile the provided PyTorch module or function for optimized execution.

        Args:
            X (Module): Function or Module to be compiled.
            backend (str): Backend for optimization. Options can be seen with `torch._dynamo.list_backends()`.

        Returns:
            Module: Compiled model.
        """
        # Determine the backend to use for compilation
        backend = (
            "inductor"
            if not backend
            or backend not in torch._dynamo.list_backends()
            or (backend == "onnxrt" and not torch.onnx.is_onnxrt_backend_supported())
            else backend
        )

        # Compile the model using the specified backend and additional options
        return torch.compile(
            model=X,
            fullgraph=True,
            backend=backend,
            options={
                "shape_padding": True,
                "triton.cudagraphs": True,
            },
        )

    def __get_model(
        self,
        weight: str,
        track: bool,
        fuse: bool,
        format: str,
        optimize: bool,
        backend: str,
        config: Dict,
        **kwargs
    ) -> partial:
        """
        Create and configure the YOLO model based on the provided parameters.

        Args:
            weight (str): Path to the YOLO model weights file.
            track (bool): Enable object tracking.
            fuse (bool): Fuse model layers for improved performance.
            format (str): Model format.
            optimize (bool): Enable model optimization using torch.compile.
            backend (str): Backend for optimization. Options can be seen with `torch._dynamo.list_backends()`.
            config (Dict): Additional configuration parameters.

        Returns:
            partial: Partially configured YOLO model.
        """
        # Create an instance of the YOLO model
        model = YOLO(weight, task="detect")

        # Fuse model layers if specified
        if fuse:
            model.fuse()

        # Optimize the model using torch.compile if specified
        if optimize:
            model = self.__compile(X=model, backend=backend)

        # Configure the track model if specified
        if track:
            model = model.track
            config.update({"persist": True, "tracker": "configs/tracker.yaml"})
        else:
            model = model.predict

        # Return a partially configured YOLO model
        return partial(model, **config, classes=0, verbose=False, **kwargs)

    def __check_half(self, half: bool) -> bool:
        """
        Check if half precision (float16) is available and applicable.

        Args:
            half: Input value indicating whether half precision should be used.

        Returns:
            bool: False if the device is on CPU; otherwise, keep the original value.
        """
        # Check if half precision is specified and the device is CPU
        if half and self.device == "cpu":
            print(
                "[yellow][WARNING] [YOLOv8]: Half is only supported on CUDA. Using default float32.[/]"
            )
            half = False
        return half

    def forward(self, image: Union[cv2.Mat, np.ndarray]) -> List[Tuple]:
        """
        Perform a forward pass of the model.

        Args:
            image (MatLike): Input image.

        Returns:
            List: A list containing detected human bounding boxes.
        """
        # Perform a forward pass of the model on the input image
        result = self.model(source=image)[0]

        outputs = []

        # Extract information from the detection results
        if result.boxes:
            for box in result.boxes:
                outputs.append(
                    {
                        "box": [int(i.item()) for i in box.xyxy[0]],
                        "score": box.conf.item(),
                        "id": int(box.id.item()) if box.is_track else None,
                    }
                )

        return outputs
