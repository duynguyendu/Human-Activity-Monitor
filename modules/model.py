from rich import print
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from lightning.pytorch import LightningModule



class LitModel(LightningModule):
    """PyTorch Lightning module"""

    def __init__(self):
        super().__init__()


    def criterion(self, y_hat, y):
        return F.cross_entropy(y_hat, y)


    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.hparams.lr)


    def _log(self, stage: str, loss, y_hat, y):
        acc = accuracy(y_hat, y, task='multiclass', num_classes=self.num_classes)
        self.log_dict(
            {f"{stage}/loss": loss, f"{stage}/accuracy": acc},
            on_step=False, on_epoch=True
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


    def save_hparams(self, config: dict):
        self.hparams.update(config)
        self.save_hyperparameters()


    def load(self, path: str, strict: bool=True, verbose: bool=True):
        if not path:
            return
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        checkpoint_state_dict = torch.load(path, map_location=device)['state_dict']
        self.load_state_dict(checkpoint_state_dict, strict=strict)
        print("[bold]Load checkpoint successfully.[/]") if verbose else None