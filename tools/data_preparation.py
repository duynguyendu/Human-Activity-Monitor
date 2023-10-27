import shutil
import hydra

# Setup root
import os, sys
sys.path.extend([os.getcwd(), f"{os.getcwd()}/src"])

from src.modules.data import DataProcessing



@hydra.main(config_path="../configs", config_name="data", version_base="1.3")
def main(cfg):
    # Remove the hydra outputs
    shutil.rmtree("outputs")

    # Config data generate
    processer = DataProcessing(**cfg['processer']) 

    # Generate data
    processer(**cfg['auto'])



if __name__=="__main__":
    main()
