from datetime import datetime
from typing import Union


class Writer:
    def __init__(self, path: str, interval: int, camera: bool) -> None:
        """
        Initialize a Writer object.

        Args:
            path (str): The base path for storing data files.
            interval (int): Save data every n (seconds).
            camera (bool): If True, use camera-related timestamps; else, use seconds.
        """
        self.path = path
        self.interval = interval
        self.camera = camera
        self.holder = {}

    def _write(self, path: str, contents: str, append: bool) -> None:
        """
        Write contents to a file specified by the given path.

        Args:
            path (str): The file path.
            contents (str): The data to be written.
            append (bool): If True, append to the file; else, overwrite the file.
        """
        with open(path, "a" if append else "w") as f:
            f.write(contents)

    def _check_format(self, data: Union[str, list]) -> str:
        """
        Check the format of data. If it's a list, convert to a comma-separated string.

        Args:
            data (Union[str, list]): The data to be formatted.

        Returns:
            str: The formatted data.
        """
        return "|".join(map(str, data)) if isinstance(data, list) else str(data)

    def _now(self) -> str:
        """
        Get the current time in HH:MM:SS format.

        Returns:
            str: The formatted current time.
        """
        return datetime.now().strftime("%H:%M:%S")

    def new(self, name: str, type: str, features: Union[str, list]) -> None:
        """
        Create a new data file for a specific data stream.

        Args:
            name (str): The name of the data stream.
            type (str): The type of data stream.
            features (Union[str, list]): The features/columns of the data stream.
        """

        # Create new holder
        self.holder[name] = {"path": f"{self.path}/{name}.{type}", "current": 0}

        # Initialize first line
        self._write(
            path=self.holder[name]["path"],
            contents=f"{'time' if self.camera else 'second'}|{self._check_format(features)}\n",
            append=False,
        )

    def has(self, name: str) -> bool:
        """
        Check if the Writer has a data stream with the given name.

        Args:
            name (str): The name of the data stream.

        Returns:
            bool: True if the data stream exists, False otherwise.
        """
        return name in self.holder

    def save(self, name: str, contents: Union[str, list], progress: int) -> None:
        """
        Save data for a specific data stream.

        Args:
            name (str): The name of the data stream.
            contents (Union[str, list]): The data to be saved.
            progress (int): The current progress of the data stream.
        """

        # Check first and duplicated progress
        if progress == 0 or progress == self.holder[name]["current"]:
            return

        # Update writer progress
        self.holder[name]["current"] = progress

        # Check save frequency
        if progress % self.interval != 0:
            return

        # Format timestamp
        timestamp = self._now() if self.camera else progress

        # Save values
        self._write(
            path=self.holder[name]["path"],
            contents=f"{timestamp}|{self._check_format(contents)}\n",
            append=True,
        )
