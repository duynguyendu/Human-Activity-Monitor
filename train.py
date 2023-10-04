from argparse import ArgumentParser
import os

from modules.callback import callbacks_list
from modules.model import LitModel
from modules.data import *
from models import *

import torch.optim as optim
import torch.nn as nn
import torch

from lightning.pytorch import seed_everything, Trainer
from rich import traceback
traceback.install()



# Set seed
seed_everything(seed=42, workers=True)

# Set number of worker (CPU will be used | Default: 60%)
NUM_WOKER = int(os.cpu_count()*0.8) if torch.cuda.is_available() else 0



def main(args):
    # Define dataset
    DATASET = CustomDataModule(
        data_path = "data/UTD-MHAD",
        sampling_value = 2,
        # max_frames = 32,
        batch_size = args.batch,
        num_workers = NUM_WOKER
    )

    # Define model
    MODEL = ViT_B_16(
        num_classes = len(DATASET.classes),
        # hidden_features = 256,
        pretrained = True,
        freeze = True
    )

    # Lightning model
    lit_model = LitModel(
        model = MODEL,
        criterion = nn.CrossEntropyLoss(),
        optimizer = optim.AdamW(MODEL.parameters(), lr=args.learning_rate),
        checkpoint = args.checkpoint
    )

    # Lightning trainer
    trainer = Trainer(
        max_epochs = args.epoch,
        precision = "16-mixed",
        callbacks = callbacks_list
    )

    # Training
    trainer.fit(lit_model, DATASET)

    # Testing
    trainer.test(lit_model, DATASET)



if __name__=="__main__":
    parser = ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=-1)
    parser.add_argument("-b", "--batch", type=int, default=None)
    parser.add_argument("-lr", "--learning_rate", type=float, default=None)
    parser.add_argument("-cp", "--checkpoint", type=str, default=None)
    args = parser.parse_args()

    main(args)
