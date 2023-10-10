from argparse import ArgumentParser
import os

from modules.scheduler import scheduler_with_warmup
from modules.callback import CustomCallbacks
from modules.model import LitModel
from modules.data import *
from models import *

import torch.optim.lr_scheduler as ls
import torch.optim as optim
import torch.nn as nn
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything, Trainer
from rich import traceback
traceback.install()



# Set seed
seed_everything(seed=42, workers=True)

# Set number of worker (CPU will be used | Default: 60%)
NUM_WOKER = int(os.cpu_count()*0.6) if torch.cuda.is_available() else 0



def main(args):
    # Define dataset
    dataset = UTD_MHADDataModule(
        data_path = "data/UTD-MHAD",
        augmentation = True,
        sampling_value = 2,
        # max_frames = 32,
        batch_size = args.batch,
        num_workers = NUM_WOKER
    )

    # Define model
    model = ViT_B_32(
        num_classes = len(dataset.classes),
        dropout = 0.0,
        attention_dropout = 0.2,
        pretrained = True,
        freeze = False
    )

    # Setup loss, optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.learning_rate)

    # Setup scheduler
    scheduler = scheduler_with_warmup(
        scheduler = ls.CosineAnnealingLR(
            optimizer = optimizer,
            T_max = args.epoch - 10
        ),
        warmup_epochs = 10,
        start_factor = 0.01
    )

    # Lightning model
    lit_model = LitModel(
        model = model,
        criterion = loss,
        optimizer = optimizer,
        scheduler = scheduler,
        checkpoint = args.checkpoint
    )

    # Lightning trainer
    trainer = Trainer(
        max_epochs = args.epoch,
        precision = "16-mixed",
        callbacks = CustomCallbacks,
        logger = TensorBoardLogger(save_dir="logs", name="new")
    )

    # Training
    trainer.fit(lit_model, dataset)

    # Testing
    trainer.test(lit_model, dataset)



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=100)
    parser.add_argument("-b", "--batch", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-cp", "--checkpoint", type=str, default=None)
    args = parser.parse_args()

    main(args)
