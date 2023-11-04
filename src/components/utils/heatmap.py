from typing import Tuple
import numpy as np
import cv2

from src.modules.utils import tuple_handler


class Heatmap:
    LAYER = None

    @staticmethod
    def new_layer_from_shape(shape: Tuple) -> np.ndarray:
        """
        Create an empty layer from the given shape

        Args:
            shape (Tuple): (height, width) of the layer

        Returns:
            np.ndarray: new empty layer be created
        """
        return np.zeros(tuple_handler(shape, max_dim=2), dtype=np.uint8)

    @staticmethod
    def new_layer_from_image(image: np.ndarray) -> np.ndarray:
        """
        Create an empty layer from the given image

        Args:
            image (np.ndarray): original image to create new layer

        Returns:
            np.ndarray: new empty layer be created
        """
        return np.zeros_like(image, dtype=np.uint8)

    @staticmethod
    def update(area: Tuple, value: int) -> np.ndarray:
        """
        Update map layer

        Args:
            area (int): area to increase
            value (_type_): amount to update
            blurriness (float, optional): the blurriness of the heat layer. Defaults to 1.0

        Returns:
            np.ndarray: layer after updated
        """

        # Grow
        x1, y1, x2, y2 = tuple_handler(area, max_dim=4)
        x1, y1 = int(x1 * 0.95), int(y1 * 0.95)
        x2, y2 = int(x2 * 1.05), int(y2 * 1.05)
        Heatmap.LAYER[y1:y2, x1:x2] = np.minimum(
            Heatmap.LAYER[y1:y2, x1:x2] + value, 255 - value
        )

    @staticmethod
    def decay(value: int = 1) -> None:
        """
        Reduce heatmap value

        Args:
            value (int): percentage to decrease
        """
        Heatmap.LAYER = ((1 - value / 100) * Heatmap.LAYER).astype("uint8")

    @staticmethod
    def apply(
        image: np.ndarray,
        blurriness: float = 1.0,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Apply heatmap

        Args:
            image (np.ndarray): image to apply heatmap
            layer (np.ndarray): layer to apply heat
            alpha (float, optional): Opacity of the heat layer. Defaults to 0.5

        Returns:
            np.ndarray: result image
        """

        # Blur
        blurriness = int(blurriness * 100)
        blurriness = blurriness + 1 if blurriness % 2 == 0 else blurriness
        Heatmap.LAYER = cv2.stackBlur(Heatmap.LAYER, (blurriness, blurriness), 0)

        # Apply heat to layer
        heatmap = cv2.applyColorMap(Heatmap.LAYER, cv2.COLORMAP_JET)

        # Combined image and heat layer
        cv2.addWeighted(heatmap, alpha, image, 1 - alpha, 0, image)

        return image, heatmap
