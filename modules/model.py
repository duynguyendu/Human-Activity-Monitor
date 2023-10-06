from rich import print
import os

from torchmetrics.functional import accuracy
import torch.optim as optim
import torch.nn as nn
import torch

from lightning.pytorch import LightningModule



class LitModel(LightningModule):
    """PyTorch Lightning module"""

    def __init__(
            self, 
            model: nn.Module,
            criterion: nn.Module = None,
            optimizer: optim.Optimizer = None,
            scheduler: optim.Optimizer = None,
            checkpoint: str = None
        ):
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
        if isinstance(self.optimizer, list):
            return self.optimizer, self.scheduler if isinstance(self.scheduler, list) else [self.scheduler]
        else:
            return [self.optimizer], [self.scheduler]


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
