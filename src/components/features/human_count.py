from collections import deque
import numpy as np


class HumanCount:
    def __init__(self, smoothness: int = 1) -> None:
        """
        Initializes a HumanCount object.

        Parameters:
            smoothness (int): The number of recent values to consider for smoothing.
                A higher value results in a smoother, but potentially slower, response.
        """
        self.history = deque([], maxlen=max(1, smoothness))

    def get_value(self) -> int:
        """
        Computes and returns the smoothed average of the historical values.

        Returns:
            int: The smoothed average of the historical values.
        """
        return int(np.mean(self.history))

    def update(self, value: int) -> None:
        """
        Updates the history with a new value and maintains the specified smoothness.

        Parameters:
            value (int): The new value to add to the history.

        Returns:
            None
        """
        self.history.append(value)
