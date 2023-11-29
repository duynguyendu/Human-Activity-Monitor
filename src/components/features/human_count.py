from collections import deque
from pathlib import Path
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
            current = (
                int(self.save_conf["count"] * self.save_conf["speed"])
                / self.save_conf["fps"]
            )
            if self.save_conf["count"] != 0:
                if (current % self.save_conf["interval"]) == 0:
                    with open(self.save_conf["save_path"], "a") as f:
                        f.write(f"{int(current)},{self.get_value()}\n")

            self.save_conf["count"] += 1

    def config_save(
        self, save_path: str, interval: int, fps: int, speed: int = 1
    ) -> None:
        """
        Save the counted value

        Args:
            save_path (str): Path to save output
            interval (int): Save every n (second)
            fps (int): Frame per second of the video
            speed (int): Video speed multiplying

        Returns:
            None
        """
        save_path = Path(save_path)

        # Create save folder
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            f.write("second,value" + "\n")

        self.save_conf = {
            "save_path": save_path,
            "count": 0,
            "interval": interval,
            "fps": fps,
            "speed": max(1, speed),
        }
