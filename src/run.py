import shutil

from omegaconf import DictConfig
import hydra

# Setup root directory
from rootutils import autosetup

from components.video import Video

autosetup()


@hydra.main(config_path="../configs", config_name="run", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Remove the hydra outputs
    shutil.rmtree("outputs")

    # Load video
    VIDEO = Video(**cfg["video"])

    VIDEO.setup_backbone(config=cfg)

    VIDEO.run()


if __name__ == "__main__":
    main()
