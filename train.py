from argparse import ArgumentParser
from rich import traceback
traceback.install()
import os

import torch
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer, seed_everything
from modules.callback import callbacks_list
from modules.data import UCF11DataModule
from models.VGG import VGG11



# Set seed
seed_everything(seed=42, workers=True)

# Set number of worker (CPU will be used | Default: 80%)
NUM_WOKER = int(os.cpu_count()*0.6) if torch.cuda.is_available() else 0


def main(args):
    # Define dataset
    dataset = UCF11DataModule(
        data_path="data/UCF11", 
        remake_data=False,
        sampling_value=6,
        num_frames=0,
        batch_size=args.batch,
        num_workers=NUM_WOKER
    )

    # Define model
    model = VGG11(num_classes=11, hidden_features=128)

    model.save_hparams({"lr": args.learning_rate})

    # Logger
    wandb_logger = WandbLogger(
        name=model._get_name(), 
        project="HAR", 
        log_model="all"
    )

    # Define trainer
    trainer = Trainer(
        max_epochs=args.epoch, 
        precision="16-mixed",
        logger=wandb_logger, 
        callbacks=callbacks_list
    )

    # Training
    trainer.fit(model, dataset)

    # Testing
    trainer.test(model, dataset)



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=-1)
    parser.add_argument("-b", "--batch", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-cp", "--checkpoint", type=str, default=None)
    args = parser.parse_args()

    main(args)
