from typing import Tuple
import numpy as np
import cv2

from src.modules.utils import tuple_handler


class Heatmap:
    def __init__(self, blurriness: float = 1.0, alpha: float = 0.5) -> None:
        """
        Heatmap initialize

        Args:
            blurriness (float, optional): the blurriness of the heat layer. Defaults to 1.0.
            alpha (float, optional): Opacity of the heat layer. Defaults to 0.5.
        """
        self.layer = None
        self.blurriness = int(
            100 * blurriness + 1 if 100 * blurriness % 2 == 0 else 100 * blurriness
        )
        self.alpha = alpha

    def update(self, area: Tuple, value) -> None:
        """
        Update map layer

        Args:
            area (int): area to increase
            value (_type_): _description_
        """
        x1, y1, x2, y2 = tuple_handler(area, max_dim=4)
        self.layer[y1:y2, x1:x2] += value
        self.layer = np.clip(self.layer, a_min=0, a_max=255 - value)

    def apply(self, image: np.ndarray):
        """
        Apply heatmap

        Args:
            image (np.ndarray): image to apply heatmap

        Returns:
            np.ndarray: result image
        """

        # Blur map
        self.layer = cv2.GaussianBlur(
            self.layer.astype("uint8"), (self.blurriness, self.blurriness), 0
        )

        # Apply heat
        heatmap = cv2.applyColorMap(self.layer, cv2.COLORMAP_JET)
        # Combined
        cv2.addWeighted(heatmap, self.alpha, image, 1 - self.alpha, 0, image)

        return image
