from lightning.pytorch import Trainer, LightningModule
import lightning.pytorch.callbacks as cb

from rich import print



class PrintTrainResult(cb.Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch

        current_lr = ", ".join([f"{optim.param_groups[0]['lr']:.2e}" for optim in trainer.optimizers])

        results = trainer.callback_metrics
        result_train = f"loss: {results['train/loss']:.4f}, acc: {results['train/accuracy']:.3f}"
        result_val = f"loss: {results['val/loss']:.4f}, acc: {results['val/accuracy']:.3f}"

        print(
            f"[bold]Epoch[/]( {epoch} ) "
            f"[bold]Lr[/]( {current_lr} ) "
            f"[bold]Train[/]({result_train}) "
            f"[bold]Val[/]({result_val})"
        )


CustomCallbacks = [
    PrintTrainResult(),
    cb.RichModelSummary(),
    cb.RichProgressBar(),
    cb.LearningRateMonitor(logging_interval='epoch'),
    cb.ModelCheckpoint(
        monitor = 'val/loss',
        save_weights_only = True,
        save_top_k = 2,
        save_last = True
    ),
    cb.EarlyStopping(
        monitor = 'val/loss',
        min_delta = 0.0001,
        patience = 10
    )
]
