from typing import Union
import os

from torch.nn import Module
from rich import print
import torch

from modules.model import LitModel
from models import ViT as vit


class ViT:
    CLASSES = ["idle", "laptop", "phone", "walk"]

    def __init__(
        self,
        checkpoint: str,
        device: str = "auto",
        half: bool = False,
        dynamo: bool = False,
    ):
        """
        Initialize the class.

        Args:
            - checkpoint (str): Path to the model checkpoint.
            - device (str, optional): Device to use ("auto", "cuda", or "cpu"). Defaults to "auto".
            - half (bool, optional): Use half-precision (float16). Defaults to False.
            - dynamo (bool, optional): Use TorchDynamo for GPU acceleration. Defaults to False.
        """
        # Check if the provided checkpoint path exists
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(checkpoint)
        self.ckpt = checkpoint

        # Determine the device based on user input or availability
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Create an instance of the LitModel using the specified architecture and checkpoint
        self.model = LitModel(
            model=vit("B_32", num_classes=len(self.CLASSES)), checkpoint=self.ckpt
        )

        # Apply half-precision if specified
        if half:
            if self.device == "cpu":
                print(
                    "[yellow][WARNING] ViT: Half is only supported on CUDA. Using default float32.[/]"
                )
                half = False
            else:
                self.model = self.__half(self.model)

        # Apply TorchDynamo compilation if specified
        if dynamo:
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
                raise ValueError(f"Input dimension must be 4. Got {X.dim()} instead.")
        return X

    def forward(self, image: torch.Tensor) -> str:
        """
        Forward pass of the model

        Args:
            image (MatLike): Input image

        Returns:
            str: Classified result
        """
        with torch.inference_mode():
            X = self.__check_dim(image).to(self.device)

            X = self.__half(X) if self.half else X

            outputs = self.model(X)
            _, pred = torch.max(outputs, 1)

        return self.CLASSES[pred.item()]
