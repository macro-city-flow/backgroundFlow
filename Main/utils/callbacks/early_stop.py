from pytorch_lightning.utilities import rank_zero_info
from utils.callbacks.base import BestEpochCallback
from pl_bolts.callbacks.printing import dicts_to_table

class EarlyStop(BestEpochCallback):
    def __init__(self, monitor='', mode='min'):
        super(EarlyStop, self).__init__(monitor=monitor, mode=mode)
        self.metrics_dict = {}
        
    def on_validation_epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
        return super().on_validation_epoch_end(trainer, pl_module)
    #TODO realize early stop and print it