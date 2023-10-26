from modules.data import DataProcessing
import shutil
import hydra



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
