import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.callbacks.base import BestEpochCallback


class PlotValidationPredictionsCallback(BestEpochCallback):
    def __init__(self, monitor='', mode='min'):
        super(PlotValidationPredictionsCallback, self).__init__(
            monitor=monitor, mode=mode)
        self._ground_truths = []
        self._predictions = []
        self._rect_width = 0.35

    def on_fit_start(self, trainer, pl_module):
        self._ground_truths.clear()
        self._predictions.clear()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        super().on_validation_batch_end(trainer, pl_module,
                                        outputs, batch, batch_idx, dataloader_idx)
        if trainer.current_epoch != self.best_epoch:
            return
        self._ground_truths.clear()
        self._predictions.clear()
        predictions, y = outputs
        #self.ground_truths.append(y[:, 0, :])
        #self.predictions.append(predictions[:, 0, :])
        self._ground_truths.append(y)
        self._predictions.append(predictions)

    def on_fit_end(self, trainer, pl_module):
        pass
