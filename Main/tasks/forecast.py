import argparse
from torch.autograd import backward
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression.mean_absolute_error import MeanAbsoluteError
import utils.metrics
import utils.losses

class forecastTask(pl.LightningModule):
    def __init__(self, model: nn.Module, loss='mse',
                 pre_len: int = 1, learning_rate: float = 1e-3, feature_dim: int =1,
                 weight_decay: float = 1.5e-3, feat_max_val:float=1,data_module:str='NS', **kwargs):
        super(forecastTask, self).__init__()
        self.save_hyperparameters()
        self._feature_dim = feature_dim
        self._model = model
        self._loss = loss
        self._pre_len = pre_len
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._feat_max_val = feat_max_val
        self._is_sequential = True if data_module == 'S' else False
        self._y_transfer = nn.Sigmoid()
        self._mean_absolute_error = MeanAbsoluteError()
        
    def forward(self, x):
        outputs=self._model(x)
        return outputs

    def shared_step(self, batch, batch_idx):
        x, y = batch
        predictions = self(x)
        y = self._y_transfer(y)
        return predictions, y
    
    def loss(self, inputs, targets):
        if self._loss == 'mse':
            return F.mse_loss(inputs,targets)
        else:
            raise  NameError("Loss not realized",self._loss)

    def training_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        loss = self.loss(predictions, y)
        self.log('train_loss',loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions, y = self.shared_step(batch, batch_idx)
        predictions = predictions
        y = y
        loss = self.loss(predictions, y)
        rmse = torch.sqrt(loss) 
        mae = torch.sum(self._mean_absolute_error(predictions,y))
        accuracy = utils.metrics.multiFeatureAccuracy(predictions, y)
        var = sum([torch.var(predictions[i]-y[i]) for i in range(len(predictions))])/len(predictions)
        metrics = {
            'val_loss': loss,
            'RMSE': rmse,
            'MAE': mae,
            'accuracy': accuracy,
            'variance':var
        }
        self.log_dict(metrics)
        return predictions.reshape(batch[1].size()), y.reshape(batch[1].size())

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(self._model.parameters(),  
                                lr=self.hyperparameters.get('learning_rate'), 
                                weight_decay=self.hyperparameters.get('weight_decay'))

    @staticmethod
    def add_task_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning-rate', '--lr', type=float, default=1e-3)
        parser.add_argument('--weight-decay', '--wd', type=float, default=1.5e-3)
        parser.add_argument('--loss', type=str, default='mse')
        return parser

    @property
    def hyperparameters(self):
        return {
            'learning_rate':self._learning_rate,
            'weight_decay':self._weight_decay
        }
