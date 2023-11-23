from typing import Dict, Tuple, Union
import os

from torch.nn import Module
from rich import print
import torch

from src.modules.data.transform import DataTransformation
from src.modules.utils import device_handler, tuple_handler


class Classifier:
    CLASSES = ["idle", "laptop", "phone", "walk"]

    def __init__(
        self,
        checkpoint: str,
        image_size: Union[int, Tuple] = 224,
        half: bool = False,
        optimize: bool = False,
        device: str = "auto",
    ):
        """
        Initialize the class.

        Args:
            - checkpoint (str): Path to the model checkpoint.
            - size (int or Tuple, optional): Input size for the model. Defaults to 224.
            - half (bool, optional): Use half-precision (float16). Defaults to False.
            - optimize (bool, optional): Use TorchDynamo for model optimization. Defaults to False.
            - device (str, optional): Device to use ("auto", "cuda", or "cpu"). Defaults to "auto".
        """
        # Check if the provided checkpoint path exists
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        self.ckpt = checkpoint

        # Determine the device based on user input or availability
        self.device = device_handler(device)

        # Defind transform
        self.transform = DataTransformation.TOPIL(
            image_size=tuple_handler(image_size, max_dim=2)
        )

        # Load model
        self.model = torch.load(self.ckpt, map_location=self.device)

        # Apply half-precision if specified
        if half:
            if self.device == "cpu":
                print(
                    "[yellow][WARNING] [Classifier]: Half is only supported on CUDA. Using default float32.[/]"
                )
                half = False
            else:
                self.model = self.__half(self.model)

        # Apply TorchDynamo compilation if specified
        if optimize:
            self.model = self.__compile(self.model)

        # Store configuration options
        self.half = half
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

    def forward(self, image: torch.Tensor) -> Dict:
        """
        Forward pass of the model

        Args:
            image (MatLike): Input image

        Returns:
            str: Classified result
        """

        # Transform
        X = self.transform(image)

        # Get result
        with torch.inference_mode():
            # Check dimension
            X = self.__check_dim(X).to(self.device)

            # Apply haft
            X = self.__half(X) if self.half else X

            outputs = self.model(X)

            _, pred = torch.max(outputs, 1)

        return self.CLASSES[pred.item()]
