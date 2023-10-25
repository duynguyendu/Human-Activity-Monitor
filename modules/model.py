from typing import Dict
import os

from torchmetrics.functional import accuracy
import torch.optim as optim
import torch.nn as nn
import torch

from lightning.pytorch import LightningModule
from rich import print



class LitModel(LightningModule):
    def __init__(
            self, 
            model: nn.Module,
            criterion: nn.Module = None,
            optimizer: optim.Optimizer | Dict = None,
            scheduler: optim.Optimizer = None,
            checkpoint: str = None
        ):
        """
        Initialize the Lightning Model.

        Args:
            model (nn.Module): The neural network model to be trained.
            criterion (nn.Module, optional): The loss function. Default: None
            optimizer (optim.Optimizer | Dict, optional): The optimizer or optimizer configuration. Default: None
            scheduler (optim.Optimizer, optional): The learning rate scheduler. Default: None
            checkpoint (str, optional): Path to a checkpoint file for model loading. Default: None
        """
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        if checkpoint:
            self.load(checkpoint)


    def forward(self, X):
        return self.model(X)


    def configure_optimizers(self):
        if not self.scheduler:
            return self.optimizer
        if not isinstance(self.optimizer, list):
            self.optimizer = [self.optimizer]
        if not isinstance(self.scheduler, list):
            self.scheduler = [self.scheduler]
        return self.optimizer, self.scheduler


    def _log(self, stage: str, loss, y_hat, y):
        acc = accuracy(
            preds = y_hat, 
            target = y, 
            task = 'multiclass', 
            num_classes = self.model.num_classes
        )
        self.log_dict(
            dictionary = {f"{stage}/loss": loss, f"{stage}/accuracy": acc},
            on_step = False, on_epoch = True
        )


    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self._log(stage="train", loss=loss, y_hat=y_hat, y=y)
        return loss


    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self._log(stage="val", loss=loss, y_hat=y_hat, y=y)


    def test_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X)
        loss = self.criterion(y_hat, y)
        self._log(stage="test", loss=loss, y_hat=y_hat, y=y)


    def load(self, path: str, strict: bool=True, verbose: bool=True):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_state_dict(
            state_dict = torch.load(path, map_location=device)['state_dict'],
            strict = strict
        )
        print("[bold]Load checkpoint:[/] Done") if verbose else None


    def save_hparams(self, config: Dict) -> None:
        self.hparams.update(config)
        self.save_hyperparameters()
