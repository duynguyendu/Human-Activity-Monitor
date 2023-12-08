from collections import deque
from datetime import datetime
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

        if hasattr(self, "save_conf"):
            # Save value
            self.save()

            # Update count
            self.count += 1

    def config_save(
        self, save_path: str, interval: int, fps: int, speed: int, camera: bool
    ) -> None:
        """
        Save the counted value

        Args:
            save_path (str): Path to save output
            interval (int): Save every n (second)
            fps (int): Frame per second of the video
            speed (int): Video speed multiplying
            camera (bool): If using camera
        """

        with open(save_path, "w") as f:
            f.write("time" if camera else "second" + ",value" + "\n")

        self.count = 0
        self.save_conf = {
            "save_path": save_path,
            "interval": interval,
            "fps": fps,
            "speed": max(1, speed),
            "camera": camera,
        }

    def save(self) -> None:
        """Save value"""

        # Calculate current
        current = int(self.count * self.save_conf["speed"]) / self.save_conf["fps"]

        # Not first, check interval
        if not (self.count != 0 and ((current % self.save_conf["interval"]) == 0)):
            return

        # Write result
        with open(self.save_conf["save_path"], "a") as f:
            time_format = (
                datetime.now().strftime("%H:%M:%S")
                if self.save_conf["camera"]
                else int(current)
            )
            f.write(f"{time_format},{self.get_value()}\n")

        # Reset count on camera
        if self.save_conf["camera"]:
            self.count = 0
