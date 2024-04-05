from typing import Dict, List, Tuple, Union
from functools import partial

from torch.nn import Module
import numpy as np
import torch
import cv2

from ultralytics import YOLO

from .utils import device_handler, check_half


class Detector:
    def __init__(
        self,
        task: str,
        weight: str = "weights/yolov8m.pt",
        conf: float = 0.25,
        iou: float = 0.7,
        size: Union[int, Tuple] = 640,
        half: bool = False,
        fuse: bool = False,
        onnx: bool = False,
        optimize: bool = False,
        backend: str = None,
        device: str = "auto",
    ):
        """
        Initialize the Yolo-v8 model

        Args:
            weight (str): Task for YOLO model to perform (detect, classify).
            weight (str, optional): Path to the YOLO model weights file. Defaults to None.
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

        # Check device
        self.device = device_handler(device)

        # Save config
        self.config = {
            "conf": conf,
            "iou": iou,
            "imgsz": size,
            "half": check_half(half, self.device),
            "device": self.device,
        }

        # Check task
        self.task = task.lower().strip()

        if self.task not in ["detect", "classify"]:
            raise ValueError('Only "detect" or "classify" task are acceptable.')

        # Setup model
        self.model = self.__setup_model(
            task=self.task,
            weight=weight,
            fuse=fuse,
            format="onnx" if onnx else "pt",
            optimize=optimize,
            backend=backend,
            config=self.config,
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

    def __onnx_model(self): ...

    def __tensorrt_model(self): ...

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

    def __setup_model(
        self,
        task: str,
        weight: str,
        fuse: bool,
        format: str,
        optimize: bool,
        backend: str,
        config: Dict,
    ) -> partial:
        """
        Create and configure the YOLO model based on the provided parameters.

        Args:
            weight (str): Task for YOLO model to perform.
            weight (str): Path to the YOLO model weights file.
            fuse (bool): Fuse model layers for improved performance.
            format (str): Model format.
            optimize (bool): Enable model optimization using torch.compile.
            backend (str): Backend for optimization. Options can be seen with `torch._dynamo.list_backends()`.
            config (Dict): Additional configuration parameters.

        Returns:
            partial: Partially configured YOLO model.
        """

        # Create an instance of the YOLO model
        model = YOLO(model=weight, task=task)

        # Fuse model layers if specified
        # if fuse:
        #     model.fuse()

        # Optimize the model using torch.compile if specified
        # if optimize:
        #     model = self.__compile(X=model, backend=backend)

        # Return a partially configured YOLO model
        return partial(
            model.predict,
            **config,
            classes=0 if self.task == "detect" else None,
            verbose=False
        )

    def forward(self, image: Union[cv2.Mat, np.ndarray]) -> np.ndarray:
        """
        Perform a forward pass of the model.

        Args:
            image (MatLike): Input image.

        Returns:
            np.ndarray: An array containing information of detected people.
        """

        # Perform a forward pass of the model on the input image
        result = self.model(source=image)[0]

        result = result.boxes if self.task == "detect" else result.probs

        # Get result
        return result.data.cpu().numpy()
