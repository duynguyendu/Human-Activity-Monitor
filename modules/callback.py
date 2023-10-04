from lightning.pytorch import Trainer, LightningModule
import lightning.pytorch.callbacks as cb
from rich import print



class PrintTrainResult(cb.Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        results = trainer.callback_metrics

        result_train = f"loss: {results['train/loss']:.4f}, acc: {results['train/accuracy']:.3f}"
        result_val = f"loss: {results['val/loss']:.4f}, acc: {results['val/accuracy']:.3f}"

        print(f"[bold]Epoch[/]( {epoch} )  [bold]Train[/]({result_train})  [bold]Val[/]({result_val})")


callbacks_list = [
    PrintTrainResult(),
    cb.RichModelSummary(),
    cb.RichProgressBar(),
    cb.LearningRateMonitor(logging_interval='epoch'),
    cb.ModelCheckpoint(
        monitor='val/loss',
        save_weights_only=True,
        save_top_k=2,
        save_last=True
    ),
    cb.EarlyStopping(
        monitor='val/loss',
        min_delta=0.0001,
        patience=5
    )
]
