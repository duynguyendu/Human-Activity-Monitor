import shutil

from omegaconf import DictConfig
import hydra

# Setup root directory
from rootutils import autosetup

autosetup()

from components import Video, Backbone


@hydra.main(config_path="../configs/run", config_name="run", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Remove the hydra outputs
    shutil.rmtree("outputs")

    # Load video
    VIDEO = Video(**cfg["video"], backbone=Backbone(config=cfg))

    if cfg["record"]:
        VIDEO.record(**cfg["record"])

    VIDEO.run()


if __name__ == "__main__":
    main()
