from lightning.pytorch import Trainer, LightningModule
import lightning.pytorch.callbacks as cb
from rich import print



class PrintTrainResult(cb.Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        results = trainer.callback_metrics
        result_train = ', '.join([
            f"loss: {results['train/loss']:.4f}",
            f"acc: {results['train/accuracy']:.3f}",
            f"f1: {results['train/f1_score']:.3f}"
        ])
        result_val = ', '.join([
            f"loss: {results['val/loss']:.4f}",
            f"acc: {results['val/accuracy']:.3f}",
            f"f1: {results['val/f1_score']:.3f}",
        ])
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
    )
]

# def callbacks_list(config):
#     cb_list = [
#         PrintTrainResult(),
#         cb.RichModelSummary(),
#         cb.RichProgressBar(),
#         cb.LearningRateMonitor(logging_interval='epoch'),
#     ]

#     cb_list.append(
#         cb.ModelCheckpoint(
#             monitor='val/loss',
#             save_weights_only=True,
#             dirpath=config['checkpoint']['dirpath'],
#             save_top_k=config['checkpoint']['save_top_k'],
#             save_last=config['checkpoint']['save_last']
#         )
#     ) if config['checkpoint']['enable'] else None

#     cb_list.append(
#         cb.EarlyStopping(
#             monitor='val/loss',
#             min_delta=0.0001,
#             patience=config['earlystopping']['patience']
#         )
#     ) if config['earlystopping']['enable'] else None

#     return cb_list