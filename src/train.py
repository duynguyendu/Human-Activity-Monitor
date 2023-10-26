import shutil

import torch.optim.lr_scheduler as ls
import torch.optim as optim
import torch.nn as nn
import torch

from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything, Trainer

import hydra
from omegaconf import DictConfig, open_dict
from rich import traceback
traceback.install()

# Setup root directory
import rootutils
rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=False)

from modules.scheduler import scheduler_with_warmup
from modules.callback import custom_callbacks
from modules.model import LitModel
from modules.data import *
from models import *



@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Remove the hydra outputs since we already have lightning logs
    shutil.rmtree("outputs")

    # Set precision
    torch.set_float32_matmul_precision('high')

    # Set seed
    if cfg['set_seed']:
        seed_everything(seed=cfg['set_seed'], workers=True)

    # Define dataset
    DATASET = CustomDataModule(
        **cfg['data'],
        batch_size = cfg['trainer']['batch_size'],
        num_workers = cfg['num_workers'] if torch.cuda.is_available() else 0
    )

    # Define model
    MODEL = ViT(
        version = "B_32",
        dropout = 0.0,
        attention_dropout = 0.0,
        num_classes = len(DATASET.classes),
        **cfg['model']
    )

    # Setup loss, optimizer
    LOSS = nn.CrossEntropyLoss()
    OPTIMIZER = optim.AdamW(MODEL.parameters(), lr=cfg['trainer']['learning_rate'], weight_decay=cfg['trainer']['learning_rate'])

    # Setup scheduler
    SCHEDULER = scheduler_with_warmup(
        scheduler = ls.CosineAnnealingLR(optimizer=OPTIMIZER, T_max=cfg['trainer']['num_epoch']),
        warmup_epochs = cfg['scheduler']['warmup_epochs'],
        start_factor = cfg['scheduler']['start_factor']
    )

    # Lightning model
    LIT_MODEL = LitModel(MODEL, LOSS, OPTIMIZER, SCHEDULER, cfg['trainer']['checkpoint'])

    # Save config
    with open_dict(cfg):
        cfg['model']['name'] = MODEL._get_name()
        if hasattr(MODEL, "version"):
            cfg['model']['version'] = MODEL.version
    LIT_MODEL.save_hparams(cfg)

    # Lightning trainer
    TRAINER = Trainer(
        max_epochs = cfg['trainer']['num_epoch'],
        precision = cfg['trainer']['precision'],
        callbacks = custom_callbacks(),
        logger = TensorBoardLogger(save_dir="logs", name="new")
    )

    # Training
    TRAINER.fit(LIT_MODEL, DATASET)

    # Testing
    TRAINER.test(LIT_MODEL, DATASET)



if __name__=="__main__":
    main()
