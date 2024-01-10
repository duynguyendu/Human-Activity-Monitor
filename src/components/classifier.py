from typing import Dict, Tuple, Union
import os

import torchvision.transforms as T
from torch.nn import Module
import torch

from .utils import *


class Classifier:
    CLASSES = ["idle", "laptop", "phone", "walk"]

    def __init__(
        self,
        weight: str,
        image_size: Union[int, Tuple] = 224,
        half: bool = False,
        optimize: bool = False,
        device: str = "auto",
    ):
        """
        Initialize the class.

        Args:
            - weight (str): Path to the model weight.
            - size (int or Tuple, optional): Input size for the model. Defaults to 224.
            - half (bool, optional): Use half-precision (float16). Defaults to False.
            - optimize (bool, optional): Use TorchDynamo for model optimization. Defaults to False.
            - device (str, optional): Device to use ("auto", "cuda", or "cpu"). Defaults to "auto".
        """

        # Check if the provided weight path exists
        if not os.path.exists(weight):
            raise FileNotFoundError(weight)

        # Determine the device based on user input or availability
        self.device = device_handler(device)

        # Defind transform
        self.transform = self.__setup_transform(image_size)

        # Load model
        self.model = torch.load(weight, map_location=self.device)

        # Check half-precision
        self.half = check_half(half, self.device)

        if self.half:
            self.model = self.__half(self.model)

        # Apply TorchDynamo compilation if specified
        if optimize:
            self.model = self.__compile(self.model)

        # Set model device
        self.model.to(self.device)

    def __call__(self, image: torch.Tensor) -> str:
        """
        Forward pass of the model

        Args:
            image (MatLike): Input image

        Returns:
            str: Classified result
        """
        return self.forward(image)

    def __setup_transform(self, image_size: Union[int, Tuple]) -> T.Compose:
        """
        Setup image transform

        Args:
            image_size (int or Tuple): Desire size of the image

        Returns:
            torchvision.transforms.Compose
        """
        return T.Compose(
            [
                T.ToPILImage(),
                T.Resize(tuple_handler(image_size, max_dim=2), antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __half(self, X: Union[torch.Tensor, Module]) -> Union[torch.Tensor, Module]:
        """
        Equivalent to X.to(torch.float16)

        Args:
            X (Union[torch.Tensor, Module]): Tensor or Module to be converted

        Returns:
            Union[torch.Tensor, Module]: Converted output
        """
        return X.half()

    def __compile(self, X: Module) -> Module:
        """
        The latest method to speed up PyTorch code

        Args:
            X (Module): Function or Module to be compiled

        Returns:
            Module: Compiled output
        """
        return torch.compile(
            model=X,
            fullgraph=True,
            backend="inductor",
            options={
                "shape_padding": True,
                "triton.cudagraphs": True,
            },
        )

    def __check_dim(self, X: torch.Tensor) -> torch.Tensor:
        """
        Check input dimension

        Args:
            X (torch.Tensor): Value to be checked

        Raises:
            ValueError: If input is invalid

        Returns:
            torch.Tensor: Tensor with dim equal 4
        """
        match X.dim():
            case 3:
                X = X.unsqueeze(0)
            case 4:
                pass
            case _:
                raise ValueError(
                    f"Input dimension must be 3 (no batch) or 4 (with batch). Got {X.dim()} instead."
                )
        return X

    @torch.inference_mode
    def forward(self, image: torch.Tensor) -> Dict:
        """
        Forward pass of the model

        Args:
            image (MatLike): Input image

        Returns:
            Dict: A dictionary contains classification result. Keys:`label` and `score`
        """

        # Transform
        X = self.transform(image)

        # Check dimension
        X = self.__check_dim(X).to(self.device)

        # Apply haft
        X = self.__half(X) if self.half else X

        outputs = self.model(X)

        outputs = torch.softmax(outputs, dim=1)

        value, pos = torch.max(outputs, dim=1)

        return {"label": self.CLASSES[pos.item()], "score": value.item()}
