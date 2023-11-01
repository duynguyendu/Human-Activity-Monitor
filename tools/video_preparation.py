import shutil
import hydra

# Setup root
import sys
import os

sys.path.extend([os.getcwd(), f"{os.getcwd()}/src"])

from src.modules.data import VideoPreparation


@hydra.main(config_path="../configs/data", config_name="video", version_base="1.3")
def main(cfg):
    # Remove the hydra outputs
    shutil.rmtree("outputs")

    # Config data generate
    processer = VideoPreparation(**cfg["processer"])

    # Generate data
    processer(**cfg["auto"])


if __name__ == "__main__":
    main()
