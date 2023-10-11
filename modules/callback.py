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


def CustomCallbacks(
        model_summary: bool = True,
        progress_bar: bool = True,
        lr_monitor: bool = True,
        checkpoint: bool = True,
        early_stopping: bool = True
    ):
    callbacks = []
    callbacks.append(PrintTrainResult())
    callbacks.append(cb.RichModelSummary()) if model_summary else None
    callbacks.append(cb.RichProgressBar()) if progress_bar else None
    callbacks.append(cb.LearningRateMonitor('epoch')) if lr_monitor else None
    callbacks.append(
        cb.ModelCheckpoint(
            monitor = 'val/loss',
            save_weights_only = True,
            save_top_k = 2,
            save_last = True
        )
    ) if checkpoint else None
    callbacks.append(
        cb.EarlyStopping(
            monitor = 'val/loss',
            min_delta = 0.0001,
            patience = 10
        )
    ) if early_stopping else None
    return callbacks
